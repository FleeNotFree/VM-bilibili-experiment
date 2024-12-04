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
    'pretrain': f"{BASE_DIR}/pretrain"
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

                    # 控制请求频率
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

                    # 请求成功后的随机延迟
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
        """初始化Chrome驱动，带重试机制"""
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

                # 新增：内存和性能优化
                self.options.add_argument('--disable-software-rasterizer')
                self.options.add_argument('--disable-features=NetworkService')
                self.options.add_argument('--disable-dev-tools')
                self.options.add_argument('--no-first-run')
                self.options.add_argument('--dns-prefetch-disable')
                self.options.add_argument('--disk-cache-size=1')
                self.options.add_argument('--media-cache-size=1')

                # 新增：超时和连接设置
                self.options.add_argument('--network-control-timeout=30')
                self.options.add_argument('--page-load-strategy=eager')

                service = webdriver.ChromeService(executable_path='/usr/local/bin/chromedriver')
                self.driver = webdriver.Chrome(service=service, options=self.options)

                # 设置更合理的超时时间
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
        """安全地等待元素加载，带有更多的错误处理"""
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
        """改进的视频观看实现"""
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

                # 加载页面前先设置较短的超时时间
                self.driver.set_page_load_timeout(30)
                self.driver.set_script_timeout(30)

                try:
                    # 使用JavaScript导航来避免某些加载问题
                    self.driver.execute_script(f"window.location.href = '{url}';")
                except TimeoutException:
                    logger.warning("页面加载超时，尝试强制停止...")
                    self.driver.execute_script("window.stop();")
                except Exception as e:
                    logger.warning(f"页面导航异常: {str(e)}")
                    continue

                # 等待页面基本元素加载
                body = self.wait_for_element_safely(By.TAG_NAME, "body", timeout=20)
                if not body:
                    raise Exception("页面加载失败")

                # 等待视频元素加载，使用更可靠的选择器
                video_element = None
                for selector in ['video', '.bilibili-player-video video', '#bilibili-player video']:
                    video_element = self.wait_for_element_safely(By.CSS_SELECTOR, selector, timeout=10)
                    if video_element:
                        break

                if not video_element:
                    raise Exception("未找到视频元素")

                # 尝试播放视频，包含错误处理
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

                # 等待指定时长，期间检查视频状态
                start_time = time.time()
                while time.time() - start_time < duration:
                    try:
                        # 检查视频是否在播放
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
                    # 重新初始化driver
                    self.close()
                    self.init_driver()
                    continue

        return False

    def close(self):
        """改进的关闭方法，确保完全清理"""
        if self.driver:
            try:
                # 尝试清理其他资源
                try:
                    self.driver.execute_script("window.localStorage.clear();")
                    self.driver.execute_script("window.sessionStorage.clear();")
                except:
                    pass

                # 关闭所有窗口
                try:
                    self.driver.close()
                except:
                    pass

                # 退出浏览器
                self.driver.quit()
                logger.info("ChromeDriver 已成功关闭")
            except Exception as e:
                logger.error(f"关闭 ChromeDriver 时出错: {str(e)}")
            finally:
                self.driver = None

class ExperimentManager:
    def __init__(self):
        try:
            # 设置CSV文件路径
            self.state_videos_csv = f"{BASE_DIR}/state.csv"
            self.non_state_videos_csv = f"{BASE_DIR}/non.csv"

            # 直接从CSV文件读取视频链接列
            self.state_videos = pd.read_csv(self.state_videos_csv)['视频链接'].tolist()
            self.non_state_videos = pd.read_csv(self.non_state_videos_csv)['视频链接'].tolist()

            # 验证视频链接
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

    def save_checkpoint(self, accounts, current_account_index, results):
        """保存检查点"""
        try:
            with self.checkpoint_lock:
                checkpoint = {
                    'current_index': current_account_index,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'results': results
                }
                checkpoint_path = f"{DIRS['checkpoints']}/checkpoint_{current_account_index}.json"
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint, f, ensure_ascii=False, indent=2)
                logger.info(f"保存检查点: {checkpoint_path}")
        except Exception as e:
            logger.error(f"保存检查点失败: {str(e)}")

    def load_latest_checkpoint(self):
        """加载最新的检查点"""
        try:
            checkpoints = sorted(Path(DIRS['checkpoints']).glob('checkpoint_*.json'))
            if not checkpoints:
                return None

            latest_checkpoint = checkpoints[-1]
            with open(latest_checkpoint, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            logger.info(f"加载检查点: {latest_checkpoint}")
            return checkpoint_data
        except Exception as e:
            logger.error(f"加载检查点失败: {str(e)}")
            return None

    def save_pretrain_video_info(self, account, video_url):
        """保存预训练视频信息"""
        try:
            # 从URL提取BV号
            bvid = video_url.split('/')[-1]

            # 获取视频详细信息
            endpoint = f"{self.api.base_url}/x/web-interface/view"
            params = {'bvid': bvid}
            video_info = self.api.request_with_retry(endpoint, params)

            if video_info and video_info.get('code') == 0:
                data = video_info['data']

                # 准备写入CSV的数据
                video_data = {
                    'username': account['username'],
                    'video_title': data.get('title', ''),
                    'bvid': bvid,
                    'up_name': data.get('owner', {}).get('name', ''),
                    'up_id': data.get('owner', {}).get('mid', ''),
                    'view_count': data.get('stat', {}).get('view', 0),
                    'danmaku_count': data.get('stat', {}).get('danmaku', 0),
                    'like_count': data.get('stat', {}).get('like', 0),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

                # 写入CSV文件
                csv_path = f"{DIRS['pretrain']}/{account['group']}_pretrain.csv"
                with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=video_data.keys())
                    if f.tell() == 0:  # 如果文件为空，写入表头
                        writer.writeheader()
                    writer.writerow(video_data)

                logger.info(f"已保存用户 {account['username']} 的预训练视频信息: {bvid}")

        except Exception as e:
            logger.error(f"保存预训练视频信息失败: {str(e)}")

    def pretrain_single_user(self, account):
        """预训练单个用户"""
        if account['group'] == "control":
            return account

        video_pool = self.state_videos if account['group'] == "state" else self.non_state_videos
        driver = None

        try:
            with self.log_lock:
                logger.info(f"开始预训练用户 {account['username']} (组: {account['group']})")

            driver = BilibiliDriver()
            sampled_videos = random.choices(video_pool, k=PRE_TRAIN_VIDEOS_PER_USER)
            videos_watched = 0

            while videos_watched < PRE_TRAIN_VIDEOS_PER_USER:
                video_url = sampled_videos[videos_watched]

                # 验证URL格式
                if not validate_video_url(video_url):
                    logger.warning(f"无效的视频URL格式: {video_url}")
                    # 尝试选择新的视频
                    remaining_videos = [v for v in video_pool if v not in sampled_videos]
                    if remaining_videos:
                        new_video = random.choice(remaining_videos)
                        sampled_videos[videos_watched] = new_video
                        continue
                    else:
                        logger.error("没有更多可用的视频")
                        break

                # 观看视频
                if driver.watch_video(video_url, duration=PRE_TRAIN_VIDEO_DURATION):
                    # 保存预训练视频信息
                    self.save_pretrain_video_info(account, video_url)

                    # 记录观看历史
                    watch_record = {
                        'url': video_url,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'duration': PRE_TRAIN_VIDEO_DURATION,
                        'success': True
                    }
                    account['watched_videos'].append(watch_record)
                    videos_watched += 1

                    if videos_watched % 5 == 0:
                        logger.info(f"用户 {account['username']} 已完成 {videos_watched}/{PRE_TRAIN_VIDEOS_PER_USER} 个视频")
                else:
                    # 视频观看失败，尝试使用新的视频
                    logger.warning(f"视频观看失败: {video_url}")
                    remaining_videos = [v for v in video_pool if v not in sampled_videos]
                    if remaining_videos:
                        new_video = random.choice(remaining_videos)
                        sampled_videos[videos_watched] = new_video
                        logger.info(f"替换为新视频: {new_video}")
                        continue
                    else:
                        logger.error("没有更多可用的视频")
                        videos_watched += 1

                # 视频间随机休息
                time.sleep(random.uniform(2, 4))

            return account

        except Exception as e:
            logger.error(f"用户 {account['username']} 预训练失败: {str(e)}")
            return account
        finally:
            if driver:
                driver.close()

    def pretrain_users_parallel(self, accounts):
        """并行预训练用户"""
        try:
            # 区分不同组的账户
            state_accounts = [acc for acc in accounts if acc['group'] == 'state']
            non_state_accounts = [acc for acc in accounts if acc['group'] == 'non-state']
            control_accounts = [acc for acc in accounts if acc['group'] == 'control']

            logger.info(f"预训练开始前账户统计:")
            logger.info(f"State组账户数: {len(state_accounts)}")
            logger.info(f"Non-state组账户数: {len(non_state_accounts)}")
            logger.info(f"Control组账户数: {len(control_accounts)}")

            # 验证账户数量
            if len(state_accounts) != len(non_state_accounts):
                raise ValueError(
                    f"State组({len(state_accounts)})和Non-state组({len(non_state_accounts)})账户数量不相等")

            completed_accounts = []
            MAX_BATCH_RETRIES = 3  # 每批次最大重试次数

            # 按批次处理账户
            batch_size_per_group = PRETRAIN_BATCH_SIZE
            for i in range(0, len(state_accounts), batch_size_per_group):
                batch_state = state_accounts[i:i + batch_size_per_group]
                batch_non_state = non_state_accounts[i:i + batch_size_per_group]
                current_batch = batch_state + batch_non_state

                batch_retry_count = 0
                batch_success = False

                while batch_retry_count < MAX_BATCH_RETRIES and not batch_success:
                    if batch_retry_count > 0:
                        # 如果是重试，先休息15-30分钟
                        sleep_time = random.uniform(900, 1800)  # 15-30分钟
                        logger.warning(
                            f"批次处理全部失败，第 {batch_retry_count} 次重试，休息 {sleep_time / 60:.1f} 分钟...")
                        time.sleep(sleep_time)

                    logger.info(f"\n开始处理第 {i // batch_size_per_group + 1} 批账户 (重试次数: {batch_retry_count}):")
                    logger.info(f"State组: {[acc['username'] for acc in batch_state]}")
                    logger.info(f"Non-state组: {[acc['username'] for acc in batch_non_state]}")

                    batch_results = []
                    # 并行处理当前批次
                    with concurrent.futures.ThreadPoolExecutor(max_workers=len(current_batch)) as executor:
                        future_to_account = {
                            executor.submit(self.pretrain_single_user, account): account
                            for account in current_batch
                        }

                        for future in concurrent.futures.as_completed(future_to_account):
                            account = future_to_account[future]
                            try:
                                trained_account = future.result()
                                if trained_account and trained_account.get('watched_videos'):
                                    batch_results.append(trained_account)
                                    logger.info(f"用户 {account['username']} 预训练完成")
                            except Exception as e:
                                logger.error(f"用户 {account['username']} 预训练失败: {str(e)}")

                    # 检查批次是否成功
                    if len(batch_results) > 0:  # 只要有成功的就添加到完成列表
                        completed_accounts.extend(batch_results)
                        batch_success = True
                    else:
                        batch_retry_count += 1
                        if batch_retry_count >= MAX_BATCH_RETRIES:
                            logger.error(f"批次处理在 {MAX_BATCH_RETRIES} 次重试后仍然失败，跳过该批次")
                            # 如果最终失败，仍然保留这些账户，但不标记为已完成训练
                            completed_accounts.extend(current_batch)

                # 批次之间休息（成功的批次之间）
                if batch_success and i + batch_size_per_group < len(state_accounts):
                    rest_time = random.uniform(15, 30)
                    logger.info(f"当前批次完成，休息 {rest_time:.1f} 秒后处理下一批...")
                    time.sleep(rest_time)

            # 添加控制组账户
            completed_accounts.extend(control_accounts)

            # 验证结果
            final_counts = {
                'state': len([acc for acc in completed_accounts if acc['group'] == 'state']),
                'non-state': len([acc for acc in completed_accounts if acc['group'] == 'non-state']),
                'control': len([acc for acc in completed_accounts if acc['group'] == 'control'])
            }

            logger.info("\n预训练完成后账户统计:")
            for group, count in final_counts.items():
                logger.info(f"- {group}组: {count}")

            return completed_accounts

        except Exception as e:
            logger.error(f"预训练过程发生错误: {str(e)}")
            raise

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
        """收集单个用户的数据，带重试机制"""
        try:
            local_results = []
            retry_count = 0
            homepage_videos = None
            driver = None

            try:
                driver = BilibiliDriver()
                self.api.set_cookies(driver.get_cookies())

                # 重试获取首页视频
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
                            logger.warning(f"用户 {account['username']} 获取首页视频出错: {str(e)}，第 {retry_count} 次重试...")
                            time.sleep(random.uniform(10, 15))
                        else:
                            logger.error(f"用户 {account['username']} 获取首页视频最终失败: {str(e)}")
                            return []

                if homepage_videos:
                    logger.info(f"用户 {account['username']} 获取到 {len(homepage_videos)} 个首页视频")

                    for i, video in enumerate(homepage_videos, 1):
                        if not video:
                            continue

                        # 处理首页视频
                        video_data = None
                        retry_count = 0
                        while retry_count < max_retries and not video_data:
                            try:
                                video_data = self.prepare_video_data(video, "homepage", account, datetime.now())
                                if video_data:
                                    local_results.append(video_data)
                                    logger.info(f"用户 {account['username']} - 处理首页第 {i}/{len(homepage_videos)} 个视频成功")
                            except Exception as e:
                                retry_count += 1
                                if retry_count < max_retries:
                                    logger.warning(f"用户 {account['username']} 处理首页视频失败: {str(e)}，第 {retry_count} 次重试...")
                                    time.sleep(random.uniform(5, 10))
                                else:
                                    logger.error(f"用户 {account['username']} 处理首页视频最终失败: {str(e)}")
                                    continue

                        # 获取并处理相关视频
                        if video.get('bvid'):
                            related_videos = None
                            retry_count = 0
                            while retry_count < max_retries and not related_videos:
                                try:
                                    related_videos = self.api.get_related_videos(video.get('bvid'))
                                    if related_videos:
                                        for related in related_videos[:10]:  # 限制相关视频数量
                                            if not related:
                                                continue
                                            related_data = self.prepare_video_data(related, "recommended", account, datetime.now())
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

                    # 更新共享结果
                    with self.results_lock:
                        shared_results.extend(local_results)

                    logger.info(f"用户 {account['username']} ({account['group']}) 数据收集完成，共收集 {len(local_results)} 条数据")
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
            # 按组分类账户
            grouped_accounts = {
                'state': [acc for acc in accounts if acc['group'] == 'state'],
                'non-state': [acc for acc in accounts if acc['group'] == 'non-state'],
                'control': [acc for acc in accounts if acc['group'] == 'control']
            }

            # 验证每组账户数量
            for group, group_accounts in grouped_accounts.items():
                if len(group_accounts) != USERS_PER_GROUP:
                    raise ValueError(f"{group} 组账户数量不正确，期望 {USERS_PER_GROUP}，实际 {len(group_accounts)}")

            # 初始化结果文件
            fieldnames = ["用户名", "性别", "用户组", "视频类型", "标题", "BV号", "播放量", "弹幕数",
                         "评论数", "收藏数", "投币数", "分享数", "点赞数", "UP主", "UP主ID", "抓取时间"]

            result_files = {
                'state': f"{DIRS['results']}/state_results.csv",
                'non-state': f"{DIRS['results']}/non_state_results.csv",
                'control': f"{DIRS['results']}/control_results.csv"
            }

            # 创建结果文件
            for file_path in result_files.values():
                if not os.path.exists(file_path):
                    with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()

            all_results = []
            batch_results = []

            # 并行处理所有账户
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

            # 按组保存结果
            grouped_results = {'state': [], 'non-state': [], 'control': []}
            for result in batch_results:
                if isinstance(result, dict) and '用户组' in result:
                    group = result['用户组']
                    grouped_results[group].append(result)

            # 写入结果文件
            for group, results_list in grouped_results.items():
                if results_list:
                    with open(result_files[group], 'a', newline='', encoding='utf-8-sig') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writerows(results_list)

            all_results.extend(batch_results)

            # 保存检查点
            self.save_checkpoint(accounts, 1, all_results)

            logger.info(f"\n数据收集完成，共处理 {len(accounts)} 个账户")
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

def run_experiment(users_per_group=1):
    """运行实验的主函数"""
    try:
        logger.info("=" * 50)
        logger.info("开始实验...")
        logger.info("=" * 50)

        # 初始化实验管理器
        manager = ExperimentManager()

        # 创建虚拟用户
        logger.info(f"创建每组 {users_per_group} 个虚拟用户")
        all_accounts = []
        for group in ['state', 'non-state', 'control']:
            logger.info(f"\n创建 {group} 组用户:")
            group_accounts = create_virtual_accounts(users_per_group, group)
            all_accounts.extend(group_accounts)
            logger.info(f"已创建 {len(group_accounts)} 个 {group} 组用户")

        # 预训练前统计
        logger.info("\n" + "=" * 20 + " 预训练前账户统计 " + "=" * 20)
        for group in ['state', 'non-state', 'control']:
            count = len([acc for acc in all_accounts if acc['group'] == group])
            logger.info(f"{group} 组: {count} 个用户")
        logger.info(f"总账户数: {len(all_accounts)}")

        # 开始预训练
        logger.info("\n" + "=" * 20 + " 开始预训练 " + "=" * 20)
        trained_accounts = manager.pretrain_users_parallel(all_accounts)

        # 验证预训练结果
        logger.info("\n" + "=" * 20 + " 预训练后账户统计 " + "=" * 20)
        for group in ['state', 'non-state', 'control']:
            count = len([acc for acc in trained_accounts if acc['group'] == group])
            logger.info(f"{group} 组: {count} 个用户")
            if count != users_per_group:
                raise ValueError(f"{group} 组账户数量不正确，期望 {users_per_group}，实际 {count}")

        # 保存观看历史
        manager.save_watch_history(trained_accounts)

        # 开始数据收集
        logger.info("\n" + "=" * 20 + " 开始数据收集 " + "=" * 20)
        results = manager.collect_data_parallel(trained_accounts)

        logger.info("\n" + "=" * 20 + " 实验完成 " + "=" * 20)
        logger.info("=" * 50 + "\n")

        return results

    except Exception as e:
        logger.error(f"实验过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    # ==== 实验基础参数 ====
    USERS_PER_GROUP = 30  # 每组账户数量（state组30个，non-state组30个，control组30个）

    # ==== 预训练阶段参数 ====
    PRETRAIN_BATCH_SIZE = 10  # 每批次处理的每组账户数（每批处理25个state + 25个non-state = 50个账户）
    PRE_TRAIN_VIDEOS_PER_USER = 10  # 每个账户预训练要看的视频数
    PRE_TRAIN_VIDEO_DURATION = 1  # 每个视频观看时长（秒）

    # 打印实验参数
    logger.info("实验参数:")
    logger.info(f"- 每组用户数: {USERS_PER_GROUP}")
    logger.info("\n预训练阶段:")
    logger.info(f"- 预训练批次大小: 每组{PRETRAIN_BATCH_SIZE}个账户（总共{PRETRAIN_BATCH_SIZE * 2}个/批）")
    logger.info(f"- 每用户观看视频数: {PRE_TRAIN_VIDEOS_PER_USER}")
    logger.info(f"- 每视频观看时长: {PRE_TRAIN_VIDEO_DURATION}秒")
    logger.info("\n数据收集阶段:")
    logger.info(f"- 并行处理所有 {USERS_PER_GROUP * 3} 个账户")

    try:
        # 运行实验
        run_experiment(users_per_group=USERS_PER_GROUP)
    except KeyboardInterrupt:
        logger.warning("实验被用户中断")
    except Exception as e:
        logger.error(f"实验运行失败: {str(e)}")
    finally:
        logger.info("实验程序结束")