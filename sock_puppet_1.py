import os
from pathlib import Path
import random
import string
import requests
import time
import csv
from datetime import datetime
import pandas as pd
import json
import logging
from logging.handlers import RotatingFileHandler
import concurrent.futures
from threading import Lock
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains

# ==== 基础配置 ====
BASE_DIR = "/home/wangziye040608/sock_puppet"
DIRS = {
    'logs': f"{BASE_DIR}/logs",
    'checkpoints': f"{BASE_DIR}/checkpoints",
    'results': f"{BASE_DIR}/results",
    'state_videos': f"{BASE_DIR}/videos/state",
    'non_state_videos': f"{BASE_DIR}/videos/non",
    'pretrain': f"{BASE_DIR}/pretrain",
    'batch_logs': f"{BASE_DIR}/logs/batches"  # 新增：批次日志目录
}

# 创建所有必要的目录
for dir_path in DIRS.values():
    os.makedirs(dir_path, exist_ok=True)


# ==== 日志设置 ====
def setup_logging():
    logger = logging.getLogger('bilibili_experiment')
    logger.setLevel(logging.INFO)

    # 文件处理器
    file_handler = RotatingFileHandler(
        f"{DIRS['logs']}/experiment.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()


# ==== 工具函数 ====
def generate_username():
    """生成随机用户名"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=8))


def create_virtual_accounts(num_accounts: int, group_type: str):
    """创建虚拟账户"""
    accounts = []
    for i in range(num_accounts):
        account = {
            'username': generate_username(),
            'sex': random.choice(['male', 'female']),
            'group': group_type.strip().lower(),
            'watched_videos': []
        }
        accounts.append(account)
        logger.info(f"创建用户: {account['username']}, 组别: {account['group']}")
    return accounts


def validate_video_url(url: str) -> bool:
    """验证B站视频URL格式"""
    if not url:
        return False
    if not url.startswith('https://www.bilibili.com/video/'):
        return False
    if len(url.split('/')) < 5:
        return False
    return True


class BatchManager:
    def __init__(self):
        self.batch_lock = Lock()
        self.current_batch = 0
        self.batch_results = {}

    def get_next_batch_number(self):
        with self.batch_lock:
            self.current_batch += 1
            return self.current_batch

    def create_batch_accounts(self, batch_size_per_group):
        """创建单个批次的账户"""
        batch_accounts = []
        for group in ['state', 'non-state', 'control']:
            group_accounts = create_virtual_accounts(batch_size_per_group, group)
            batch_accounts.extend(group_accounts)
        return batch_accounts

    def wait_between_batches(self, min_interval, max_interval):
        """批次间等待"""
        wait_time = random.uniform(min_interval * 60, max_interval * 60)
        logger.info(f"等待 {wait_time / 60:.2f} 分钟后开始下一批次...")
        time.sleep(wait_time)


class PreTrainManager:
    def __init__(self):
        self.blank_page = """
        <html>
            <body style="background-color: white;">
                <h1 style="text-align: center; color: #ccc; margin-top: 40vh;">
                    休息中...
                </h1>
            </body>
        </html>
        """

    def show_blank_page(self, driver):
        """显示空白页"""
        try:
            blank_page_path = f"{DIRS['logs']}/blank.html"
            with open(blank_page_path, 'w', encoding='utf-8') as f:
                f.write(self.blank_page)
            driver.get(f"file://{blank_page_path}")
        except Exception as e:
            logger.error(f"显示空白页失败: {str(e)}")

    def pretrain_group(self, accounts, video_pool, videos_per_user, videos_per_group, video_duration):
        """并行预训练一组用户"""
        try:
            # 过滤出需要预训练的账户（非control组）
            train_accounts = [acc for acc in accounts if acc['group'] != 'control']

            # 为每个账户准备视频列表
            for account in train_accounts:
                account['training_videos'] = random.sample(video_pool, videos_per_user)
                account['video_groups'] = [
                    account['training_videos'][i:i + videos_per_group]
                    for i in range(0, len(account['training_videos']), videos_per_group)
                ]

            video_groups_count = videos_per_user // videos_per_group
            # 对每个视频组进行训练
            for group_idx in range(video_groups_count):
                logger.info(f"开始处理第 {group_idx + 1}/{video_groups_count} 组视频")

                # 并行处理所有账户的当前视频组
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(train_accounts)) as executor:
                    futures = {
                        executor.submit(
                            self.process_video_group,
                            account,
                            group_idx,
                            video_duration
                        ): account for account in train_accounts
                    }

                    for future in concurrent.futures.as_completed(futures):
                        account = futures[future]
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"账户 {account['username']} 视频组 {group_idx + 1} 处理失败: {str(e)}")

                # 该组视频看完后休息
                rest_time = (10 + group_idx) * 60  # 递增休息时间（分钟）
                logger.info(f"第 {group_idx + 1} 组视频完成，休息 {rest_time / 60} 分钟")
                time.sleep(rest_time)

            return train_accounts

        except Exception as e:
            logger.error(f"预训练组失败: {str(e)}")
            raise

    def process_video_group(self, account, group_idx, video_duration):
        """处理单个账户的一组视频"""
        driver = None
        try:
            driver = BilibiliDriver()
            videos = account['video_groups'][group_idx]

            for video in videos:
                if driver.watch_video(video, duration=video_duration):
                    # 记录观看历史
                    watch_record = {
                        'url': video,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'duration': video_duration,
                        'success': True
                    }
                    account['watched_videos'].append(watch_record)

                # 视频间短暂休息
                time.sleep(random.uniform(2, 4))

            # 显示空白页
            self.show_blank_page(driver)
            return True

        except Exception as e:
            logger.error(f"处理视频组失败 (用户: {account['username']}, 组: {group_idx}): {str(e)}")
            return False
        finally:
            if driver:
                driver.close()

class BilibiliAPI:
    def __init__(self):
        self.base_url = "https://api.bilibili.com"
        self.session = requests.Session()
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edge/91.0.864.59',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605.1.15'
        ]
        self.request_lock = Lock()
        self.last_request_time = time.time()
        self.error_count = 0
        self.MAX_ERROR_COUNT = 5

    def set_cookies(self, cookies):
        """设置session cookies"""
        if cookies:
            for cookie in cookies:
                self.session.cookies.set(cookie['name'], cookie['value'])

    def _get_random_ua(self):
        """获取随机User-Agent"""
        return random.choice(self.user_agents)

    def _should_reset_error_count(self):
        """检查是否需要重置错误计数"""
        if self.error_count >= self.MAX_ERROR_COUNT:
            logger.warning(f"连续错误次数达到{self.MAX_ERROR_COUNT}次，进入长时间休息...")
            time.sleep(random.uniform(300, 600))  # 休息5-10分钟
            self.error_count = 0
            return True
        return False

    def request_with_retry(self, endpoint, params=None, max_retries=5):
        """发送API请求，带重试机制"""
        with self.request_lock:
            self._should_reset_error_count()

            for attempt in range(max_retries):
                try:
                    headers = {
                        'User-Agent': self._get_random_ua(),
                        'Referer': 'https://www.bilibili.com',
                        'Accept': 'application/json, text/plain, */*',
                        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                        'Connection': 'keep-alive'
                    }

                    current_time = time.time()
                    time_since_last = current_time - self.last_request_time
                    if time_since_last < 2:
                        time.sleep(2 - time_since_last)

                    response = requests.get(
                        endpoint,
                        params=params,
                        headers=headers,
                        timeout=30
                    )

                    if response.status_code == 412:
                        logger.warning("触发反爬机制，等待较长时间...")
                        time.sleep(random.uniform(500, 600))
                        continue

                    response.raise_for_status()
                    self.last_request_time = time.time()
                    self.error_count = 0

                    time.sleep(random.uniform(5, 8))

                    return response.json()

                except Exception as e:
                    self.error_count += 1
                    logger.warning(f"请求失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")

                    if attempt < max_retries - 1:
                        delay = (2 ** attempt * 30) + random.uniform(10, 30)
                        logger.info(f"等待 {delay:.2f} 秒后重试...")
                        time.sleep(delay)
                    else:
                        logger.error(f"达到最大重试次数，请求失败")
                        raise

    def get_homepage_videos(self):
        """获取首页推荐视频"""
        try:
            endpoint = f"{self.base_url}/x/web-interface/index/top/feed/rcmd"
            params = {"pn": 1}
            data = self.request_with_retry(endpoint, params)

            if data.get("code") == 0 and "data" in data:
                return data["data"]["item"]
            else:
                logger.error(f"API返回错误: {data.get('message', '未知错误')}")
                return None

        except Exception as e:
            logger.error(f"获取首页视频失败: {str(e)}")
            return None

    def get_related_videos(self, bvid: str):
        """获取相关视频推荐"""
        try:
            if not bvid:
                return None

            endpoint = f"{self.base_url}/x/web-interface/archive/related"
            params = {"bvid": bvid}
            data = self.request_with_retry(endpoint, params)

            if data.get("code") == 0 and "data" in data:
                return data["data"]
            else:
                logger.error(f"API返回错误: {data.get('message', '未知错误')}")
                return None

        except Exception as e:
            logger.error(f"获取相关视频失败: {str(e)}")
            return None

class BilibiliDriver:
    def __init__(self):
        self.driver = None
        self.cookies = None
        self.init_driver()

    def init_driver(self, max_retries=3):
        """初始化Chrome驱动"""
        for attempt in range(max_retries):
            try:
                self.options = webdriver.ChromeOptions()
                # 基础配置
                self.options.add_argument('--headless')
                self.options.add_argument('--no-sandbox')
                self.options.add_argument('--disable-dev-shm-usage')
                self.options.add_argument('--disable-gpu')
                self.options.add_argument('--window-size=1920,1080')

                # 性能优化配置
                self.options.add_argument('--disable-extensions')
                self.options.add_argument('--disable-notifications')
                self.options.add_argument('--disable-default-apps')
                self.options.add_argument('--disable-popup-blocking')

                # 内存和性能优化
                self.options.add_argument('--disable-software-rasterizer')
                self.options.add_argument('--disable-features=NetworkService')
                self.options.add_argument('--disable-dev-tools')
                self.options.add_argument('--no-first-run')
                self.options.add_argument('--dns-prefetch-disable')
                self.options.add_argument('--disk-cache-size=1')
                self.options.add_argument('--media-cache-size=1')

                # 超时和连接设置
                self.options.add_argument('--network-control-timeout=30')
                self.options.add_argument('--page-load-strategy=eager')

                service = webdriver.ChromeService(executable_path='/usr/local/bin/chromedriver')
                self.driver = webdriver.Chrome(service=service, options=self.options)

                self.driver.set_page_load_timeout(30)
                self.driver.set_script_timeout(30)
                self.driver.implicitly_wait(10)

                self.driver.maximize_window()

                logger.info("ChromeDriver 初始化成功")
                return True

            except Exception as e:
                logger.error(f"ChromeDriver 初始化失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if self.driver:
                    try:
                        self.driver.quit()
                    except:
                        pass
                    self.driver = None

                if attempt == max_retries - 1:
                    raise
                time.sleep(5)
        return False

    def wait_for_element_safely(self, by, value, timeout=20):
        """安全地等待元素加载"""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
            return element
        except TimeoutException:
            logger.warning(f"等待元素 {value} 超时")
            return None
        except Exception as e:
            logger.warning(f"等待元素 {value} 失败: {str(e)}")
            return None

    def watch_video(self, url, duration=1, max_retries=3):
        """观看视频"""
        for attempt in range(max_retries):
            try:
                if not self.driver:
                    self.init_driver()

                logger.info(f"尝试加载视频: {url}")

                # 清理cookies和缓存
                self.driver.delete_all_cookies()
                try:
                    self.driver.execute_script("window.localStorage.clear();")
                    self.driver.execute_script("window.sessionStorage.clear();")
                except:
                    pass

                self.driver.set_page_load_timeout(30)
                self.driver.set_script_timeout(30)

                try:
                    self.driver.execute_script(f"window.location.href = '{url}';")
                except TimeoutException:
                    logger.warning("页面加载超时，尝试强制停止...")
                    self.driver.execute_script("window.stop();")
                except Exception as e:
                    logger.warning(f"页面导航异常: {str(e)}")
                    continue

                body = self.wait_for_element_safely(By.TAG_NAME, "body", timeout=20)
                if not body:
                    raise Exception("页面加载失败")

                video_element = None
                for selector in ['video', '.bilibili-player-video video', '#bilibili-player video']:
                    video_element = self.wait_for_element_safely(By.CSS_SELECTOR, selector, timeout=10)
                    if video_element:
                        break

                if not video_element:
                    raise Exception("未找到视频元素")

                try:
                    self.driver.execute_script("""
                        const video = document.querySelector('video');
                        if (video) {
                            video.muted = true;
                            video.currentTime = 0;
                            var playPromise = video.play();
                            if (playPromise !== undefined) {
                                playPromise.catch(error => {
                                    console.log('视频播放失败:', error);
                                });
                            }
                        }
                    """)
                except Exception as e:
                    logger.warning(f"视频播放脚本执行失败: {str(e)}")

                start_time = time.time()
                while time.time() - start_time < duration:
                    try:
                        is_playing = self.driver.execute_script("""
                            const video = document.querySelector('video');
                            return video && !video.paused && video.currentTime > 0;
                        """)
                        if not is_playing:
                            logger.warning("视频未在播放，尝试重新播放...")
                            self.driver.execute_script("document.querySelector('video').play()")
                    except:
                        pass
                    time.sleep(0.5)

                logger.info(f"成功观看视频 {duration} 秒")
                return True

            except Exception as e:
                logger.error(f"观看视频失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")

                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    self.close()
                    self.init_driver()
                    continue

        return False

    def close(self):
        """关闭浏览器"""
        if self.driver:
            try:
                try:
                    self.driver.execute_script("window.localStorage.clear();")
                    self.driver.execute_script("window.sessionStorage.clear();")
                except:
                    pass

                try:
                    self.driver.close()
                except:
                    pass

                self.driver.quit()
                logger.info("ChromeDriver 已成功关闭")
            except Exception as e:
                logger.error(f"关闭 ChromeDriver 时出错: {str(e)}")
            finally:
                self.driver = None

class ExperimentManager:
    def __init__(self):
        try:
            self.state_videos_csv = f"{BASE_DIR}/state.csv"
            self.non_state_videos_csv = f"{BASE_DIR}/non.csv"

            self.state_videos = pd.read_csv(self.state_videos_csv)['视频链接'].tolist()
            self.non_state_videos = pd.read_csv(self.non_state_videos_csv)['视频链接'].tolist()

            self.state_videos = [url for url in self.state_videos if validate_video_url(url)]
            self.non_state_videos = [url for url in self.non_state_videos if validate_video_url(url)]

            if not self.state_videos:
                raise ValueError("未找到有效的state视频链接")
            if not self.non_state_videos:
                raise ValueError("未找到有效的non-state视频链接")

            self.api = BilibiliAPI()
            self.log_lock = Lock()
            self.results_lock = Lock()
            self.checkpoint_lock = Lock()

            logger.info(f"已加载 {len(self.state_videos)} 个state视频和 {len(self.non_state_videos)} 个non-state视频")

        except Exception as e:
            logger.error(f"初始化视频池失败: {str(e)}")
            raise

    def save_checkpoint(self, accounts, current_batch, results):
        """保存检查点"""
        try:
            with self.checkpoint_lock:
                checkpoint = {
                    'current_batch': current_batch,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'results': results
                }
                checkpoint_path = f"{DIRS['checkpoints']}/checkpoint_{current_batch}.json"
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint, f, ensure_ascii=False, indent=2)
                logger.info(f"保存检查点: {checkpoint_path}")
        except Exception as e:
            logger.error(f"保存检查点失败: {str(e)}")

    def prepare_video_data(self, video, video_type, user, current_time):
        """处理视频数据"""
        try:
            return {
                "用户名": user['username'],
                "性别": user['sex'],
                "用户组": user['group'],
                "视频类型": video_type,
                "标题": video.get('title', ''),
                "BV号": video.get('bvid', ''),
                "播放量": video.get('stat', {}).get('view', 0),
                "弹幕数": video.get('stat', {}).get('danmaku', 0),
                "评论数": video.get('stat', {}).get('reply', 0),
                "收藏数": video.get('stat', {}).get('favorite', 0),
                "投币数": video.get('stat', {}).get('coin', 0),
                "分享数": video.get('stat', {}).get('share', 0),
                "点赞数": video.get('stat', {}).get('like', 0),
                "UP主": video.get('owner', {}).get('name', ''),
                "UP主ID": video.get('owner', {}).get('mid', ''),
                "抓取时间": current_time.strftime("%Y-%m-%d %H:%M"),
            }
        except Exception as e:
            logger.error(f"处理视频数据时出错: {str(e)}")
            return None

    def collect_data_for_user(self, account, shared_results, max_retries=3):
        """收集单个用户的数据"""
        try:
            local_results = []
            retry_count = 0
            homepage_videos = None
            driver = None

            try:
                driver = BilibiliDriver()
                self.api.set_cookies(driver.get_cookies())

                while retry_count < max_retries and not homepage_videos:
                    try:
                        homepage_videos = self.api.get_homepage_videos()
                        if homepage_videos:
                            break
                        retry_count += 1
                        if retry_count < max_retries:
                            logger.warning(f"用户 {account['username']} 获取首页视频失败，第 {retry_count} 次重试...")
                            time.sleep(random.uniform(10, 15))
                    except Exception as e:
                        retry_count += 1
                        if retry_count < max_retries:
                            logger.warning(
                                f"用户 {account['username']} 获取首页视频出错: {str(e)}，第 {retry_count} 次重试...")
                            time.sleep(random.uniform(10, 15))
                        else:
                            logger.error(f"用户 {account['username']} 获取首页视频最终失败: {str(e)}")
                            return []

                if homepage_videos:
                    logger.info(f"用户 {account['username']} 获取到 {len(homepage_videos)} 个首页视频")

                    for i, video in enumerate(homepage_videos, 1):
                        if not video:
                            continue

                        video_data = None
                        retry_count = 0
                        while retry_count < max_retries and not video_data:
                            try:
                                video_data = self.prepare_video_data(video, "homepage", account, datetime.now())
                                if video_data:
                                    local_results.append(video_data)
                                    logger.info(
                                        f"用户 {account['username']} - 处理首页第 {i}/{len(homepage_videos)} 个视频成功")
                            except Exception as e:
                                retry_count += 1
                                if retry_count < max_retries:
                                    logger.warning(
                                        f"用户 {account['username']} 处理首页视频失败: {str(e)}，第 {retry_count} 次重试...")
                                    time.sleep(random.uniform(5, 10))
                                else:
                                    logger.error(f"用户 {account['username']} 处理首页视频最终失败: {str(e)}")
                                    continue

                        if video.get('bvid'):
                            related_videos = None
                            retry_count = 0
                            while retry_count < max_retries and not related_videos:
                                try:
                                    related_videos = self.api.get_related_videos(video.get('bvid'))
                                    if related_videos:
                                        for related in related_videos[:10]:
                                            if not related:
                                                continue
                                            related_data = self.prepare_video_data(related, "recommended", account,
                                                                                   datetime.now())
                                            if related_data:
                                                local_results.append(related_data)
                                    break
                                except Exception as e:
                                    retry_count += 1
                                    if retry_count < max_retries:
                                        logger.warning(f"获取相关视频失败: {str(e)}，第 {retry_count} 次重试...")
                                        time.sleep(random.uniform(5, 10))
                                    else:
                                        logger.error(f"获取相关视频最终失败: {str(e)}")

                    with self.results_lock:
                        shared_results.extend(local_results)

                    logger.info(
                        f"用户 {account['username']} ({account['group']}) 数据收集完成，共收集 {len(local_results)} 条数据")
                    return local_results
                else:
                    logger.error(f"用户 {account['username']} 获取首页视频失败")
                    return []

            finally:
                if driver:
                    driver.close()

        except Exception as e:
            logger.error(f"收集用户 {account['username']} 数据时出错: {str(e)}")
            return []

    def collect_data_parallel(self, accounts):
        """并行收集所有用户数据"""
        try:
            grouped_accounts = {
                'state': [acc for acc in accounts if acc['group'] == 'state'],
                'non-state': [acc for acc in accounts if acc['group'] == 'non-state'],
                'control': [acc for acc in accounts if acc['group'] == 'control']
            }

            fieldnames = ["用户名", "性别", "用户组", "视频类型", "标题", "BV号", "播放量", "弹幕数",
                          "评论数", "收藏数", "投币数", "分享数", "点赞数", "UP主", "UP主ID", "抓取时间"]

            result_files = {
                'state': f"{DIRS['results']}/state_results.csv",
                'non-state': f"{DIRS['results']}/non_state_results.csv",
                'control': f"{DIRS['results']}/control_results.csv"
            }

            for file_path in result_files.values():
                if not os.path.exists(file_path):
                    with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()

            all_results = []
            batch_results = []

            logger.info(f"开始并行处理所有 {len(accounts)} 个账户的数据收集")
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(accounts)) as executor:
                future_to_account = {
                    executor.submit(self.collect_data_for_user, account, batch_results): account
                    for account in accounts
                }

                completed = 0
                for future in concurrent.futures.as_completed(future_to_account):
                    account = future_to_account[future]
                    try:
                        future.result()
                        completed += 1
                        logger.info(f"完成账户处理: {completed}/{len(accounts)}")
                    except Exception as e:
                        logger.error(f"处理账户 {account['username']} 失败: {str(e)}")

            grouped_results = {'state': [], 'non-state': [], 'control': []}
            for result in batch_results:
                if isinstance(result, dict) and '用户组' in result:
                    group = result['用户组']
                    grouped_results[group].append(result)

            for group, results_list in grouped_results.items():
                if results_list:
                    with open(result_files[group], 'a', newline='', encoding='utf-8-sig') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writerows(results_list)

            all_results.extend(batch_results)
            return all_results

        except Exception as e:
            logger.error(f"数据收集过程出错: {str(e)}")
            raise

    def save_watch_history(self, accounts):
        """保存观看历史"""
        try:
            watch_history_path = f"{DIRS['results']}/watch_history.csv"
            with open(watch_history_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['username', 'group', 'video_url', 'watch_timestamp', 'duration'])
                for account in accounts:
                    for video in account.get('watched_videos', []):
                        writer.writerow([
                            account['username'],
                            account['group'],
                            video.get('url', ''),
                            video.get('timestamp', ''),
                            video.get('duration', '')
                        ])
            logger.info(f"观看历史已保存至: {watch_history_path}")
        except Exception as e:
            logger.error(f"保存观看历史失败: {str(e)}")


def run_batch_experiment(batch_number, total_batches, batch_size_per_group, videos_per_user,
                         videos_per_group, video_duration):
    """运行单个批次的实验"""
    try:
        logger.info(f"开始批次 {batch_number}/{total_batches}")

        # 创建批次账户
        batch_manager = BatchManager()
        accounts = batch_manager.create_batch_accounts(batch_size_per_group)

        # 初始化实验管理器
        experiment_manager = ExperimentManager()
        pretrain_manager = PreTrainManager()

        # 执行预训练
        trained_accounts = pretrain_manager.pretrain_group(
            accounts,
            experiment_manager.state_videos,
            videos_per_user,
            videos_per_group,
            video_duration
        )

        # 执行实验数据收集
        results = experiment_manager.collect_data_parallel(trained_accounts)

        # 保存观看历史和结果
        experiment_manager.save_watch_history(trained_accounts)

        # 记录批次完成信息
        batch_log_path = f"{DIRS['batch_logs']}/batch_{batch_number}.json"
        batch_info = {
            'batch_number': batch_number,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'accounts_processed': len(trained_accounts),
            'results_collected': len(results)
        }
        with open(batch_log_path, 'w', encoding='utf-8') as f:
            json.dump(batch_info, f, ensure_ascii=False, indent=2)

        logger.info(f"批次 {batch_number} 完成")
        return results

    except Exception as e:
        logger.error(f"批次 {batch_number} 执行失败: {str(e)}")
        raise


def main(total_users_per_group, batch_size_per_group, videos_per_user,
         videos_per_group, video_duration, min_batch_interval, max_batch_interval):
    """主函数"""
    try:
        total_batches = total_users_per_group // batch_size_per_group

        logger.info("=" * 50)
        logger.info("开始大规模实验...")
        logger.info(f"总用户数: {total_users_per_group * 3}")
        logger.info(f"总批次数: {total_batches}")
        logger.info(f"每批次用户数: {batch_size_per_group * 3}")
        logger.info(f"每用户视频数: {videos_per_user}")
        logger.info(f"每组视频数: {videos_per_group}")
        logger.info(f"批次间隔: {min_batch_interval}-{max_batch_interval}分钟")
        logger.info("=" * 50)

        batch_manager = BatchManager()
        all_results = []

        for batch in range(1, total_batches + 1):
            try:
                batch_results = run_batch_experiment(
                    batch,
                    total_batches,
                    batch_size_per_group,
                    videos_per_user,
                    videos_per_group,
                    video_duration
                )
                all_results.extend(batch_results)

                if batch < total_batches:
                    batch_manager.wait_between_batches(min_batch_interval, max_batch_interval)

            except Exception as e:
                logger.error(f"批次 {batch} 失败: {str(e)}")
                continue

        logger.info("\n" + "=" * 20 + " 实验完成 " + "=" * 20)
        logger.info(f"总处理批次: {total_batches}")
        logger.info(f"总收集数据条数: {len(all_results)}")
        logger.info("=" * 50 + "\n")

    except Exception as e:
        logger.error(f"实验过程中出错: {str(e)}")
        raise


if __name__ == "__main__":
    # ==== 实验参数设置 ====
    TOTAL_USERS_PER_GROUP = 900   # 每组总用户数（state/non-state/control各900个）
    BATCH_SIZE_PER_GROUP = 15     # 每批次每组用户数（每批次共45个用户）
    VIDEOS_PER_USER = 100         # 每个用户要观看的视频总数
    VIDEOS_PER_GROUP = 10         # 每组视频数（分10组，每组10个视频）
    VIDEO_DURATION = 1            # 每个视频观看时长（秒）
    MIN_BATCH_INTERVAL = 28       # 批次间最小间隔（分钟）
    MAX_BATCH_INTERVAL = 33       # 批次间最大间隔（分钟）

    # 打印实验参数
    logger.info("\n" + "=" * 20 + " 实验参数 " + "=" * 20)
    logger.info(f"- 每组总用户数: {TOTAL_USERS_PER_GROUP}")
    logger.info(f"- 每批次每组用户数: {BATCH_SIZE_PER_GROUP}")
    logger.info(f"- 总批次数: {TOTAL_USERS_PER_GROUP // BATCH_SIZE_PER_GROUP}")
    logger.info(f"- 每用户视频总数: {VIDEOS_PER_USER}")
    logger.info(f"- 视频分组数: {VIDEOS_PER_USER // VIDEOS_PER_GROUP}")
    logger.info(f"- 每组视频数: {VIDEOS_PER_GROUP}")
    logger.info(f"- 视频观看时长: {VIDEO_DURATION}秒")
    logger.info(f"- 批次间隔: {MIN_BATCH_INTERVAL}-{MAX_BATCH_INTERVAL}分钟")
    logger.info("=" * 50 + "\n")

    try:
        # 运行实验
        main(
            total_users_per_group=TOTAL_USERS_PER_GROUP,
            batch_size_per_group=BATCH_SIZE_PER_GROUP,
            videos_per_user=VIDEOS_PER_USER,
            videos_per_group=VIDEOS_PER_GROUP,
            video_duration=VIDEO_DURATION,
            min_batch_interval=MIN_BATCH_INTERVAL,
            max_batch_interval=MAX_BATCH_INTERVAL
        )
    except KeyboardInterrupt:
        logger.warning("\n实验被用户中断")
    except Exception as e:
        logger.error(f"\n实验运行失败: {str(e)}")
    finally:
        logger.info("\n实验程序结束")