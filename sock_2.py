import os
import shutil
from pathlib import Path
import random
import string
import requests
import time
import csv
from datetime import datetime, timedelta
from collections import defaultdict
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
import threading

# ==== 基础配置 ====
BASE_DIR = "/home/carl_zhou/sock_puppet"
DIRS = {
    'logs': f"{BASE_DIR}/logs",
    'checkpoints': f"{BASE_DIR}/checkpoints",
    'results': f"{BASE_DIR}/results",
    'state_videos': f"{BASE_DIR}/videos/state",
    'non_state_videos': f"{BASE_DIR}/videos/non",
    'pretrain': f"{BASE_DIR}/pretrain",
    'batch_logs': f"{BASE_DIR}/logs/batches",
    'cookies': f"{BASE_DIR}/cookies"
}


class DirectoryManager:
    @staticmethod
    def clean_directories():
        """清空所有工作目录"""
        for dir_path in DIRS.values():
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"清理并重建目录: {dir_path}")
            except Exception as e:
                logger.error(f"清理目录失败 {dir_path}: {str(e)}")

    @staticmethod
    def ensure_directories():
        """确保所有必要的目录存在"""
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


# 初始化日志
logger = setup_logging()


class ResourceManager:
    def __init__(self):
        self.active_drivers = set()
        self.driver_lock = Lock()

    def cleanup_old_logs(self, days=7):
        """清理超过指定天数的日志文件"""
        try:
            cutoff = datetime.now() - timedelta(days=days)
            for log_dir in [DIRS['logs'], DIRS['batch_logs']]:
                for file in os.listdir(log_dir):
                    file_path = os.path.join(log_dir, file)
                    if os.path.isfile(file_path):
                        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        if file_time < cutoff:
                            os.remove(file_path)
                            logger.info(f"已删除过期日志文件: {file_path}")
        except Exception as e:
            logger.error(f"清理日志文件失败: {str(e)}")

    def cleanup_old_checkpoints(self, keep_last=5):
        """只保留最近的N个检查点文件"""
        try:
            checkpoint_files = sorted(
                [f for f in os.listdir(DIRS['checkpoints']) if f.startswith('checkpoint_')],
                key=lambda x: os.path.getmtime(os.path.join(DIRS['checkpoints'], x)),
                reverse=True
            )

            for checkpoint in checkpoint_files[keep_last:]:
                file_path = os.path.join(DIRS['checkpoints'], checkpoint)
                os.remove(file_path)
                logger.info(f"已删除旧检查点文件: {file_path}")
        except Exception as e:
            logger.error(f"清理检查点文件失败: {str(e)}")

    def register_driver(self, driver):
        """注册新的driver实例"""
        with self.driver_lock:
            self.active_drivers.add(driver)

    def unregister_driver(self, driver):
        """注销driver实例"""
        with self.driver_lock:
            self.active_drivers.discard(driver)

    def cleanup_all_drivers(self):
        """清理所有活动的driver实例"""
        with self.driver_lock:
            for driver in self.active_drivers:
                try:
                    driver.close()
                except Exception as e:
                    logger.error(f"清理driver失败: {str(e)}")
            self.active_drivers.clear()


class ExperimentMonitor:
    def __init__(self):
        self.start_time = None
        self.stats = {
            'total_accounts': 0,
            'successful_accounts': 0,
            'failed_accounts': 0,
            'total_videos': 0,
            'successful_videos': 0,
            'failed_videos': 0,
            'errors': defaultdict(int)
        }
        self.stats_lock = Lock()

    def start_experiment(self):
        """开始实验计时"""
        self.start_time = datetime.now()

    def record_error(self, error_type, error_msg):
        """记录错误信息"""
        with self.stats_lock:
            self.stats['errors'][f"{error_type}: {error_msg}"] += 1

    def update_stats(self, **kwargs):
        """更新统计信息"""
        with self.stats_lock:
            for key, value in kwargs.items():
                if key in self.stats:
                    self.stats[key] += value

    def generate_report(self):
        """生成实验报告"""
        if not self.start_time:
            return "实验未开始"

        duration = datetime.now() - self.start_time
        success_rate = (self.stats['successful_accounts'] / self.stats['total_accounts'] * 100
                        if self.stats['total_accounts'] > 0 else 0)

        report = f"""
实验统计报告
====================
运行时间: {duration}
账户统计:
- 总账户数: {self.stats['total_accounts']}
- 成功账户数: {self.stats['successful_accounts']}
- 失败账户数: {self.stats['failed_accounts']}
- 成功率: {success_rate:.2f}%

视频统计:
- 总视频数: {self.stats['total_videos']}
- 成功观看数: {self.stats['successful_videos']}
- 失败观看数: {self.stats['failed_videos']}
- 视频成功率: {(self.stats['successful_videos'] / self.stats['total_videos'] * 100) if self.stats['total_videos'] > 0 else 0:.2f}%

错误统计:
"""
        for error, count in self.stats['errors'].items():
            report += f"- {error}: {count}次\n"

        return report

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

                    # 请求间隔控制
                    current_time = time.time()
                    time_since_last = current_time - self.last_request_time
                    if time_since_last < 2:
                        time.sleep(2 - time_since_last)

                    response = self.session.get(
                        endpoint,
                        params=params,
                        headers=headers,
                        timeout=45
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
            params = {
                'ps': 30,  # 每页数量
                'fresh_type': 3,
                'fresh_idx': random.randint(1, 100),
                'fresh_idx_1h': random.randint(1, 100),
                'feed_version': 'V8'
            }
            response = self.request_with_retry(endpoint, params)
            if response and 'data' in response:
                return response['data'].get('item', [])
            return []
        except Exception as e:
            logger.error(f"获取首页视频失败: {str(e)}")
            return []

    def get_related_videos(self, bvid):
        """获取相关视频推荐"""
        try:
            endpoint = f"{self.base_url}/x/web-interface/archive/related"
            params = {'bvid': bvid}
            response = self.request_with_retry(endpoint, params)
            if response and 'data' in response:
                return response['data']
            return []
        except Exception as e:
            logger.error(f"获取相关视频失败: {str(e)}")
            return []

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
                self.options.add_argument('--headless=new')  # 新版本的headless模式
                self.options.add_argument('--no-sandbox')
                self.options.add_argument('--disable-dev-shm-usage')
                self.options.add_argument('--disable-gpu')
                self.options.add_argument('--window-size=1920,1080')

                # 添加这些配置来避免一些常见错误
                self.options.add_argument('--disable-extensions')
                self.options.add_argument('--disable-notifications')
                self.options.binary_location = '/usr/bin/google-chrome'  # 指定Chrome路径
                self.options.add_argument('--remote-debugging-port=9222')  # 添加调试端口

                service = webdriver.ChromeService(executable_path='/usr/local/bin/chromedriver')
                self.driver = webdriver.Chrome(service=service, options=self.options)

                self.driver.set_page_load_timeout(45)
                self.driver.set_script_timeout(45)
                self.driver.implicitly_wait(15)

                self.driver.maximize_window()
                logger.info("ChromeDriver initialized successfully")
                return True

            except Exception as e:
                logger.error(f"ChromeDriver initialization failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
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

    def get(self, url):
        """访问URL"""
        try:
            if not self.driver:
                self.init_driver()
            self.driver.get(url)
            return True
        except Exception as e:
            logger.error(f"访问URL失败 {url}: {str(e)}")
            return False

    def add_cookie(self, cookie):
        """添加cookie"""
        try:
            if self.driver:
                self.driver.add_cookie(cookie)
        except Exception as e:
            logger.error(f"添加cookie失败: {str(e)}")

    def wait_for_element_safely(self, by, value, timeout=30):
        """安全地等待元素加载"""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
            return element
        except TimeoutException:
            logger.warning(f"等待元素超时: {value}")
            return None
        except Exception as e:
            logger.warning(f"等待元素失败 {value}: {str(e)}")
            return None

    def watch_video(self, url, duration=1, timeout=60, max_retries=3):
        """观看视频"""
        for attempt in range(max_retries):
            try:
                if not self.driver:
                    self.init_driver()

                logger.info(f"尝试加载视频: {url}")

                # 设置页面加载超时
                self.driver.set_page_load_timeout(timeout)
                self.driver.set_script_timeout(timeout)

                # 添加页面加载计时和刷新机制
                page_load_start = time.time()
                page_load_timeout = 60
                try:
                    self.driver.execute_script(f"window.location.href = '{url}';")
                except TimeoutException:
                    if time.time() - page_load_start > page_load_timeout:
                        logger.warning("页面加载超时，尝试刷新...")
                        self.driver.refresh()
                        time.sleep(2)
                    else:
                        logger.warning("页面加载超时，尝试强制停止...")
                        self.driver.execute_script("window.stop();")
                except Exception as e:
                    logger.warning(f"页面导航异常: {str(e)}")
                    continue

                # 等待和检查页面主体
                body = self.wait_for_element_safely(By.TAG_NAME, "body", timeout=timeout)
                if not body:
                    if time.time() - page_load_start > page_load_timeout:
                        logger.warning("页面响应超时，尝试刷新...")
                        self.driver.refresh()
                        time.sleep(2)
                        body = self.wait_for_element_safely(By.TAG_NAME, "body", timeout=timeout)
                    if not body:
                        raise Exception("页面加载失败")

                # 查找视频元素
                video_element = None
                video_find_start = time.time()
                video_find_timeout = 60

                for selector in ['video', '.bilibili-player-video video', '#bilibili-player video']:
                    video_element = self.wait_for_element_safely(By.CSS_SELECTOR, selector, timeout=10)
                    if video_element:
                        break
                    # 如果查找时间过长则刷新
                    if time.time() - video_find_start > video_find_timeout:
                        logger.warning("查找视频元素超时，尝试刷新...")
                        self.driver.refresh()
                        time.sleep(2)
                        video_find_start = time.time()

                if not video_element:
                    raise Exception("未找到视频元素")

                # 尝试播放视频
                try:
                    self.driver.execute_script("""
                        var video = document.querySelector('video');
                        if (video) {
                            video.muted = true;
                            video.currentTime = 0;
                            video.play();
                        }
                    """)
                except Exception as e:
                    logger.warning(f"视频播放脚本执行失败: {str(e)}")

                # 监控播放状态
                play_start_time = time.time()
                play_timeout = duration + 10  # 额外10秒缓冲

                while time.time() - play_start_time < play_timeout:
                    try:
                        is_playing = self.driver.execute_script("""
                            var video = document.querySelector('video');
                            if (!video) return false;
                            if (video.paused) {
                                video.play();
                                return false;
                            }
                            return video.currentTime > 0;
                        """)
                        if not is_playing:
                            logger.warning("视频未在播放，尝试重新播放...")
                        elif time.time() - play_start_time >= duration:
                            break
                    except Exception as e:
                        logger.warning(f"检查播放状态失败: {str(e)}")
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
                logger.info("ChromeDriver已成功关闭")
            except Exception as e:
                logger.error(f"关闭ChromeDriver时出错: {str(e)}")
            finally:
                self.driver = None

class BatchManager:
    def __init__(self):
        self.batch_lock = Lock()
        self.current_batch = 0
        self.batch_results = {}

    def get_next_batch_number(self):
        """获取下一个批次号"""
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
        self.training_lock = Lock()
        self.batch_completion_count = 0
        self.batch_completion_lock = Lock()
        self.batch_event = threading.Event()

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
        try:
            # 保留所有账户，但只对非control组进行训练
            state_accounts = sum(1 for acc in accounts if acc['group'] == 'state')
            non_state_accounts = sum(1 for acc in accounts if acc['group'] == 'non-state')
            control_accounts = sum(1 for acc in accounts if acc['group'] == 'control')
            total_accounts = state_accounts + non_state_accounts  # 不包含control组

            logger.info(f"开始预训练: state组 {state_accounts}个账户, non-state组 {non_state_accounts}个账户")

            # 为所有账户准备视频列表
            for account in accounts:
                if account['group'] != 'control':
                    account['training_videos'] = random.sample(video_pool, videos_per_user)
                    account['video_groups'] = [
                        account['training_videos'][i:i + videos_per_group]
                        for i in range(0, len(account['training_videos']), videos_per_group)
                    ]
                account['completed_videos_count'] = 0

            video_groups_count = videos_per_user // videos_per_group

            # 对每个视频组进行训练
            for group_idx in range(video_groups_count):
                logger.info(f"\n=== 开始处理第 {group_idx + 1}/{video_groups_count} 组视频 ===")

                # 重置批次完成计数和事件
                self.batch_completion_count = 0
                self.batch_event.clear()

                batch_target = videos_per_group
                batch_drivers = {}

                with concurrent.futures.ThreadPoolExecutor(max_workers=total_accounts) as executor:
                    futures = []
                    for account in accounts:
                        if account['group'] != 'control':
                            future = executor.submit(
                                self.process_video_batch,
                                account,
                                group_idx,
                                video_duration,
                                total_videos=videos_per_user,
                                batch_target=batch_target,
                                total_accounts=total_accounts,
                                batch_drivers=batch_drivers
                            )
                            futures.append(future)

                    concurrent.futures.wait(futures)

                logger.info(f"批次 {group_idx + 1} 所有账户已完成目标数量 {batch_target}")

                if group_idx < video_groups_count - 1:
                    rest_time = (10 + group_idx) * 60
                    logger.info(f"\n=== 第 {group_idx + 1} 组视频完成，开始休息 {rest_time / 60} 分钟 ===")
                    time.sleep(rest_time)

            return accounts

        except Exception as e:
            logger.error(f"预训练组失败: {str(e)}")
            raise

    def process_video_batch(self, account, group_idx, video_duration, total_videos, batch_target, total_accounts,
                            batch_drivers):
        driver = None
        try:
            driver = BilibiliDriver()
            batch_drivers[account['username']] = driver

            cookies = load_cookies(account['username'])
            if cookies:
                for cookie in cookies:
                    driver.add_cookie(cookie)

            videos_watched_in_batch = 0
            # 创建这个组的视频池副本
            available_videos = account['video_groups'][group_idx].copy()

            while videos_watched_in_batch < batch_target:
                if not available_videos:  # 如果所有视频都试过了，重新填充视频池
                    available_videos = account['video_groups'][group_idx].copy()
                    logger.info(f"账户 {account['username']} 重置视频池")

                # 从剩余视频中随机选择一个
                video = random.choice(available_videos)
                # 从可用池中移除这个视频
                available_videos.remove(video)

                logger.info(
                    f"账户 {account['username']} 开始观看第 {account['completed_videos_count'] + 1}/{total_videos} 个视频: {video}")

                if driver.watch_video(video, duration=video_duration, timeout=60):
                    # 只有成功才记录
                    with self.training_lock:
                        account['completed_videos_count'] += 1
                        videos_watched_in_batch += 1
                        watch_record = {
                            'url': video,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'duration': video_duration,
                            'video_number': account['completed_videos_count'],
                            'success': True
                        }
                        account['watched_videos'].append(watch_record)
                        logger.info(
                            f"账户 {account['username']} 完成第 {account['completed_videos_count']}/{total_videos} 个视频")
                else:
                    logger.warning(f"账户 {account['username']} 观看视频失败: {video}")

                time.sleep(random.uniform(2, 4))

            self.show_blank_page(driver)
            self._increment_batch_completion(total_accounts)
            self.batch_event.wait()

            return True

        except Exception as e:
            logger.error(f"处理视频批次失败 (用户: {account['username']}, 组: {group_idx}): {str(e)}")
            return False
        finally:
            if driver:
                driver.close()

    def _increment_batch_completion(self, total_accounts):
        """增加批次完成计数，当所有账户都完成时设置事件"""
        with self.batch_completion_lock:
            self.batch_completion_count += 1
            logger.info(f"当前批次已完成账户数: {self.batch_completion_count}/{total_accounts}")
            if self.batch_completion_count >= total_accounts:
                self.batch_event.set()

class ExperimentManager:
    def __init__(self):
        try:
            self.state_videos_csv = f"{BASE_DIR}/state.csv"
            self.non_state_videos_csv = f"{BASE_DIR}/non.csv"

            # 检查文件存在性
            if not os.path.exists(self.state_videos_csv):
                raise FileNotFoundError(f"找不到state视频文件: {self.state_videos_csv}")
            if not os.path.exists(self.non_state_videos_csv):
                raise FileNotFoundError(f"找不到non-state视频文件: {self.non_state_videos_csv}")

            # 读取视频列表
            try:
                self.state_videos = pd.read_csv(self.state_videos_csv)['视频链接'].tolist()
            except UnicodeDecodeError:
                self.state_videos = pd.read_csv(self.state_videos_csv, encoding='gbk')['视频链接'].tolist()

            try:
                self.non_state_videos = pd.read_csv(self.non_state_videos_csv)['视频链接'].tolist()
            except UnicodeDecodeError:
                self.non_state_videos = pd.read_csv(self.non_state_videos_csv, encoding='gbk')['视频链接'].tolist()

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

    def save_checkpoint(self, accounts, current_batch, results):
        """保存检查点"""
        try:
            with self.checkpoint_lock:
                checkpoint = {
                    'current_batch': current_batch,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'accounts': accounts,
                    'results': results
                }
                checkpoint_path = f"{DIRS['checkpoints']}/checkpoint_{current_batch}.json"
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint, f, ensure_ascii=False, indent=2)
                logger.info(f"保存检查点: {checkpoint_path}")
        except Exception as e:
            logger.error(f"保存检查点失败: {str(e)}")

    def collect_data_parallel(self, accounts, max_workers=None):
        """并行收集数据"""
        shared_results = []
        max_workers = max_workers or min(len(accounts), 10)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.collect_data_for_user, account, shared_results): account
                for account in accounts
            }

            for future in concurrent.futures.as_completed(futures):
                account = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"收集用户 {account['username']} 数据失败: {str(e)}")

        return shared_results

    def save_watch_history(self, accounts):
        """保存观看历史"""
        try:
            watch_history_path = f"{DIRS['results']}/watch_history.csv"
            with open(watch_history_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['username', 'group', 'video_url', 'watch_timestamp', 'duration',
                                 'video_number', 'success'])
                for account in accounts:
                    for video in account.get('watched_videos', []):
                        writer.writerow([
                            account['username'],
                            account['group'],
                            video.get('url', ''),
                            video.get('timestamp', ''),
                            video.get('duration', ''),
                            video.get('video_number', ''),
                            video.get('success', False)
                        ])
            logger.info(f"观看历史已保存至: {watch_history_path}")
        except Exception as e:
            logger.error(f"保存观看历史失败: {str(e)}")

    def collect_data_for_user(self, account, shared_results, max_retries=3):
        """收集单个用户的数据"""
        try:
            local_results = []
            retry_count = 0
            homepage_videos = None
            driver = None

            try:
                driver = BilibiliDriver()
                # 加载用户cookie
                cookies = load_cookies(account['username'])
                if cookies:
                    for cookie in cookies:
                        driver.add_cookie(cookie)

                self.api.set_cookies(cookies)

                # 获取首页视频
                for attempt in range(max_retries):
                    try:
                        homepage_videos = self.api.get_homepage_videos()
                        if homepage_videos:
                            break
                        logger.warning(f"用户 {account['username']} 第 {attempt + 1} 次获取首页视频失败")
                        if attempt < max_retries - 1:
                            time.sleep(random.uniform(10, 15))
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        logger.warning(f"获取首页视频出错: {str(e)}，第 {attempt + 1} 次重试...")
                        time.sleep(random.uniform(10, 15))

                if not homepage_videos:
                    raise Exception("无法获取首页视频")

                # 处理每个视频
                for video in homepage_videos:
                    video_data = {
                        'username': account['username'],
                        'group': account['group'],
                        'video_url': video.get('uri', ''),
                        'video_title': video.get('title', ''),
                        'author': video.get('owner', {}).get('name', ''),
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'source': 'homepage'
                    }
                    local_results.append(video_data)

                    # 获取相关推荐视频
                    if video.get('bvid'):
                        related_videos = self.api.get_related_videos(video.get('bvid'))
                        if related_videos:
                            for related in related_videos[:10]:  # 限制相关视频数量
                                related_data = {
                                    'username': account['username'],
                                    'group': account['group'],
                                    'video_url': related.get('uri', ''),
                                    'video_title': related.get('title', ''),
                                    'author': related.get('owner', {}).get('name', ''),
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'source': 'recommended',
                                    'from_video': video.get('bvid')
                                }
                                local_results.append(related_data)

                # 添加到共享结果
                with self.results_lock:
                    shared_results.extend(local_results)

                logger.info(
                    f"用户 {account['username']} ({account['group']}) 数据收集完成，收集 {len(local_results)} 条数据")
                return local_results

            except Exception as e:
                logger.error(f"收集用户 {account['username']} 数据时出错: {str(e)}")
                return []
            finally:
                if driver:
                    driver.close()

        except Exception as e:
            logger.error(f"处理用户 {account['username']} 失败: {str(e)}")
            return []


def init_account_cookies():
    """初始化账户cookies"""
    try:
        driver = BilibiliDriver()
        # 先访问目标网站
        if driver.get("https://www.bilibili.com"):
            # 等待页面加载完成
            time.sleep(2)

            cookies = driver.driver.get_cookies()
            # 验证cookie有效性
            if cookies:
                for cookie in cookies:
                    if not all(k in cookie for k in ['name', 'value', 'domain']):
                        logger.error(f"Cookie格式不完整: {cookie}")
                        return None
                    # 确保domain是bilibili.com相关
                    if not cookie['domain'].endswith('bilibili.com'):
                        logger.error(f"无效的cookie domain: {cookie['domain']}")
                        return None

            driver.close()
            return cookies if cookies else None
        return None
    except Exception as e:
        logger.error(f"初始化cookies失败: {str(e)}")
        return None

def validate_video_url(url: str) -> bool:
    """验证B站视频URL格式"""
    if not url:
        return False
    if not url.startswith('https://www.bilibili.com/video/'):
        return False
    if len(url.split('/')) < 5:
        return False
    return True

def generate_username():
    """生成随机用户名"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=8))

def save_cookies(username, cookies):
    """保存账户的cookies"""
    try:
        cookie_path = os.path.join(DIRS['cookies'], f"{username}.json")
        with open(cookie_path, 'w', encoding='utf-8') as f:
            json.dump(cookies, f, ensure_ascii=False)
        logger.info(f"Cookies已保存: {username}")
    except Exception as e:
        logger.error(f"保存Cookies失败 ({username}): {str(e)}")

def load_cookies(username):
    """加载账户的cookies"""
    try:
        cookie_path = os.path.join(DIRS['cookies'], f"{username}.json")
        if os.path.exists(cookie_path):
            with open(cookie_path, 'r', encoding='utf-8') as f:
                cookies = json.load(f)
            logger.info(f"Cookies已加载: {username}")
            return cookies
        return None
    except Exception as e:
        logger.error(f"加载Cookies失败 ({username}): {str(e)}")
        return None


def create_virtual_accounts(num_accounts: int, group_type: str):
    """创建虚拟账户直到达到目标数量"""
    accounts = []
    attempt = 0
    while len(accounts) < num_accounts:
        attempt += 1
        try:
            username = generate_username()
            cookies = init_account_cookies()

            if not cookies:
                logger.warning(f"第 {attempt} 次尝试: 创建账户cookies失败，将重试")
                time.sleep(5)
                continue

            save_cookies(username, cookies)
            loaded_cookies = load_cookies(username)  # 验证cookies是否成功保存和加载

            if not loaded_cookies:
                logger.warning(f"第 {attempt} 次尝试: 账户cookies保存或加载失败，将重试")
                time.sleep(2)
                continue

            account = {
                'username': username,
                'sex': random.choice(['male', 'female']),
                'group': group_type.strip().lower(),
                'watched_videos': [],
                'completed_videos_count': 0
            }
            accounts.append(account)
            logger.info(
                f"第 {attempt} 次尝试: 成功创建用户 {account['username']}, 组别: {account['group']}, 进度: {len(accounts)}/{num_accounts}")

        except Exception as e:
            logger.error(f"第 {attempt} 次尝试: 创建账户出错: {str(e)}")
            time.sleep(2)
            continue

    logger.info(f"完成 {group_type} 组账户创建，共尝试 {attempt} 次，成功创建 {len(accounts)} 个账户")
    return accounts


def run_batch_experiment(batch_number, total_batches, batch_size_per_group, videos_per_user,
                         videos_per_group, video_duration):
    """运行单个批次的实验"""
    try:
        logger.info(f"\n{'=' * 20} 开始批次 {batch_number}/{total_batches} {'=' * 20}")

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

        # 验证预训练结果
        incomplete_accounts = []
        for account in trained_accounts:
            if account['group'] != 'control' and account['completed_videos_count'] < videos_per_user:
                incomplete_accounts.append(account['username'])
                logger.warning(
                    f"账户 {account['username']} 预训练未完成: {account['completed_videos_count']}/{videos_per_user}")

        if incomplete_accounts:
            logger.warning(f"批次 {batch_number} 中有 {len(incomplete_accounts)} 个账户预训练未完成")

        # 执行实验数据收集
        results = experiment_manager.collect_data_parallel(trained_accounts)

        # 保存观看历史
        experiment_manager.save_watch_history(trained_accounts)

        # 保存检查点
        experiment_manager.save_checkpoint(trained_accounts, batch_number, results)

        # 记录批次完成信息
        batch_log_path = f"{DIRS['batch_logs']}/batch_{batch_number}.json"
        batch_info = {
            'batch_number': batch_number,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'accounts_processed': len(trained_accounts),
            'results_collected': len(results),
            'incomplete_accounts': incomplete_accounts,
            'success_rate': (len(trained_accounts) - len(incomplete_accounts)) / len(
                trained_accounts) if trained_accounts else 0
        }
        with open(batch_log_path, 'w', encoding='utf-8') as f:
            json.dump(batch_info, f, ensure_ascii=False, indent=2)

        logger.info(f"\n{'=' * 20} 批次 {batch_number} 完成 {'=' * 20}")
        return results

    except Exception as e:
        logger.error(f"批次 {batch_number} 执行失败: {str(e)}")
        raise


def main():
    # ==== 实验参数设置 ====
    TOTAL_USERS_PER_GROUP = 10  # 每组总用户数（state/non-state/control各900个）
    BATCH_SIZE_PER_GROUP = 10  # 每批次每组用户数（每批次共45个用户）
    VIDEOS_PER_USER = 100  # 每个用户要观看的视频总数
    VIDEOS_PER_GROUP = 10  # 每组视频数（分10组，每组10个视频）
    VIDEO_DURATION = 30  # 每个视频观看时长（秒）
    MIN_BATCH_INTERVAL = 28  # 批次间最小间隔（分钟）
    MAX_BATCH_INTERVAL = 33  # 批次间最大间隔（分钟）


    try:
        # 验证参数
        if VIDEOS_PER_USER % VIDEOS_PER_GROUP != 0:
            raise ValueError(f"每用户视频数({VIDEOS_PER_USER})必须是每组视频数({VIDEOS_PER_GROUP})的整数倍")
        if TOTAL_USERS_PER_GROUP % BATCH_SIZE_PER_GROUP != 0:
            raise ValueError(f"每组总用户数({TOTAL_USERS_PER_GROUP})必须是每批次用户数({BATCH_SIZE_PER_GROUP})的整数倍")

        # 清理所有目录
        logger.info("开始清理工作目录...")
        DirectoryManager.clean_directories()

        # 确保目录存在
        DirectoryManager.ensure_directories()

        # 初始化资源管理器和监控器
        resource_manager = ResourceManager()
        experiment_monitor = ExperimentMonitor()
        experiment_monitor.start_experiment()

        total_batches = TOTAL_USERS_PER_GROUP // BATCH_SIZE_PER_GROUP

        logger.info("\n" + "=" * 50)
        logger.info("开始大规模实验...")
        logger.info(f"总用户数: {TOTAL_USERS_PER_GROUP * 3}")
        logger.info(f"总批次数: {total_batches}")
        logger.info(f"每批次用户数: {BATCH_SIZE_PER_GROUP * 3}")
        logger.info(f"每用户视频数: {VIDEOS_PER_USER}")
        logger.info(f"每组视频数: {VIDEOS_PER_GROUP}")
        logger.info(f"视频观看时长: {VIDEO_DURATION}秒")
        logger.info(f"批次间隔: {MIN_BATCH_INTERVAL}-{MAX_BATCH_INTERVAL}分钟")
        logger.info("=" * 50)

        batch_manager = BatchManager()
        all_results = []
        batch_success_count = 0

        for batch in range(1, total_batches + 1):
            try:
                batch_results = run_batch_experiment(
                    batch,
                    total_batches,
                    BATCH_SIZE_PER_GROUP,
                    VIDEOS_PER_USER,
                    VIDEOS_PER_GROUP,
                    VIDEO_DURATION
                )

                if batch_results:
                    all_results.extend(batch_results)
                    batch_success_count += 1

                # 更新监控统计
                experiment_monitor.update_stats(
                    total_accounts=BATCH_SIZE_PER_GROUP * 3,
                    successful_accounts=len(batch_results) if batch_results else 0,
                    failed_accounts=BATCH_SIZE_PER_GROUP * 3 - (len(batch_results) if batch_results else 0)
                )

                if batch < total_batches:
                    batch_manager.wait_between_batches(MIN_BATCH_INTERVAL, MAX_BATCH_INTERVAL)

            except Exception as e:
                logger.error(f"批次 {batch} 失败: {str(e)}")
                experiment_monitor.record_error("BatchError", str(e))
                continue

        # 清理旧文件
        resource_manager.cleanup_old_logs()
        resource_manager.cleanup_old_checkpoints()

        # 生成并保存最终报告
        final_report = experiment_monitor.generate_report()
        report_path = f"{DIRS['results']}/final_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(final_report)

        logger.info("\n" + "=" * 20 + " 实验完成 " + "=" * 20)
        logger.info(f"总处理批次: {total_batches}")
        logger.info(f"成功批次数: {batch_success_count}")
        logger.info(f"总收集数据条数: {len(all_results)}")
        logger.info(f"最终报告已保存至: {report_path}")
        logger.info("=" * 50 + "\n")

    except Exception as e:
        logger.error(f"实验过程中出错: {str(e)}")
        experiment_monitor.record_error("ExperimentError", str(e))
        raise
    finally:
        # 确保清理所有资源
        resource_manager.cleanup_all_drivers()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n实验被用户中断")
    except Exception as e:
        logger.error(f"\n实验运行失败: {str(e)}")
    finally:
        logger.info("\n实验程序结束")