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

# ==== 基础配置 ====
BASE_DIR = "/home/wangziye040608/sock_puppet"
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

# ==== 实验参数设置 ====
EXPERIMENT_CONFIG = {
    'TOTAL_USERS_PER_GROUP': 900,  # 每组总用户数
    'BATCH_SIZE_PER_GROUP': 15,    # 每批次每组用户数
    'VIDEOS_PER_USER': 20,         # 每个用户要观看的视频总数
    'VIDEOS_PER_GROUP': 10,        # 每组视频数
    'VIDEO_DURATION': 1,           # 每个视频观看时长（秒）
    'MIN_BATCH_INTERVAL': 28,      # 批次间最小间隔（分钟）
    'MAX_BATCH_INTERVAL': 33,      # 批次间最大间隔（分钟）
    'MAX_RETRIES': 3,              # 最大重试次数
    'TIMEOUT': 45,                 # 默认超时时间（秒）
    'MAX_WORKERS': 10              # 最大并发工作线程数
}

# ==== 全局变量 ====
global_resource_manager = None
global_experiment_monitor = None


class DirectoryManager:
    @staticmethod
    def validate_directories():
        """验证目录结构并确保必要的目录存在"""
        for dir_path in DIRS.values():
            try:
                os.makedirs(dir_path, exist_ok=True)
            except Exception as e:
                raise RuntimeError(f"无法创建目录 {dir_path}: {str(e)}")

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


class ResourceManager:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ResourceManager, cls).__new__(cls)
                cls._instance.initialized = False
            return cls._instance

    def __init__(self):
        if not self.initialized:
            self.active_drivers = set()
            self.driver_lock = Lock()
            self.initialized = True

    def register_driver(self, driver):
        """注册新的driver实例"""
        with self.driver_lock:
            self.active_drivers.add(driver)
            logger.debug(f"已注册新的driver实例，当前活动driver数: {len(self.active_drivers)}")

    def unregister_driver(self, driver):
        """注销driver实例"""
        with self.driver_lock:
            self.active_drivers.discard(driver)
            logger.debug(f"已注销driver实例，当前活动driver数: {len(self.active_drivers)}")

    def cleanup_all_drivers(self):
        """清理所有活动的driver实例"""
        with self.driver_lock:
            for driver in list(self.active_drivers):
                try:
                    driver.close()
                    self.active_drivers.discard(driver)
                except Exception as e:
                    logger.error(f"清理driver失败: {str(e)}")
            self.active_drivers.clear()

    def cleanup_old_files(self, logs_days=7, checkpoints_keep=5):
        """清理旧文件"""
        try:
            # 清理旧日志
            cutoff = datetime.now() - timedelta(days=logs_days)
            for log_dir in [DIRS['logs'], DIRS['batch_logs']]:
                for file in os.listdir(log_dir):
                    file_path = os.path.join(log_dir, file)
                    if os.path.isfile(file_path):
                        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        if file_time < cutoff:
                            os.remove(file_path)
                            logger.info(f"已删除过期日志文件: {file_path}")

            # 清理旧检查点
            checkpoint_files = sorted(
                [f for f in os.listdir(DIRS['checkpoints']) if f.startswith('checkpoint_')],
                key=lambda x: os.path.getmtime(os.path.join(DIRS['checkpoints'], x)),
                reverse=True
            )
            for checkpoint in checkpoint_files[checkpoints_keep:]:
                file_path = os.path.join(DIRS['checkpoints'], checkpoint)
                os.remove(file_path)
                logger.info(f"已删除旧检查点文件: {file_path}")

        except Exception as e:
            logger.error(f"清理旧文件失败: {str(e)}")


class ExperimentMonitor:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ExperimentMonitor, cls).__new__(cls)
                cls._instance.initialized = False
            return cls._instance

    def __init__(self):
        if not self.initialized:
            self.start_time = None
            self.stats = {
                'total_accounts': 0,
                'successful_accounts': 0,
                'failed_accounts': 0,
                'total_videos': 0,
                'successful_videos': 0,
                'failed_videos': 0,
                'total_batches': 0,
                'successful_batches': 0,
                'failed_batches': 0,
                'errors': defaultdict(int)
            }
            self.stats_lock = Lock()
            self.initialized = True

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

        with self.stats_lock:
            account_success_rate = (self.stats['successful_accounts'] / self.stats['total_accounts'] * 100
                                    if self.stats['total_accounts'] > 0 else 0)
            video_success_rate = (self.stats['successful_videos'] / self.stats['total_videos'] * 100
                                  if self.stats['total_videos'] > 0 else 0)
            batch_success_rate = (self.stats['successful_batches'] / self.stats['total_batches'] * 100
                                  if self.stats['total_batches'] > 0 else 0)

            report = f"""
实验统计报告
====================
运行时间: {duration}

批次统计:
- 总批次数: {self.stats['total_batches']}
- 成功批次数: {self.stats['successful_batches']}
- 失败批次数: {self.stats['failed_batches']}
- 批次成功率: {batch_success_rate:.2f}%

账户统计:
- 总账户数: {self.stats['total_accounts']}
- 成功账户数: {self.stats['successful_accounts']}
- 失败账户数: {self.stats['failed_accounts']}
- 账户成功率: {account_success_rate:.2f}%

视频统计:
- 总视频数: {self.stats['total_videos']}
- 成功观看数: {self.stats['successful_videos']}
- 失败观看数: {self.stats['failed_videos']}
- 视频成功率: {video_success_rate:.2f}%

错误统计:
"""
            for error, count in self.stats['errors'].items():
                report += f"- {error}: {count}次\n"

            return report


def setup_logging():
    """设置日志系统"""
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

# 初始化全局资源管理器
global_resource_manager = ResourceManager()
global_experiment_monitor = ExperimentMonitor()


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
            if self._should_reset_error_count():
                self.error_count = 0

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
                        timeout=EXPERIMENT_CONFIG['TIMEOUT']
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
                'ps': 30,
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
        if self.driver:
            global_resource_manager.register_driver(self)

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
                self.options.add_argument('--network-control-timeout=45')
                self.options.add_argument('--page-load-strategy=eager')

                service = webdriver.ChromeService(executable_path='/usr/local/bin/chromedriver')
                self.driver = webdriver.Chrome(service=service, options=self.options)

                self.driver.set_page_load_timeout(EXPERIMENT_CONFIG['TIMEOUT'])
                self.driver.set_script_timeout(EXPERIMENT_CONFIG['TIMEOUT'])
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

    def watch_video(self, url, duration=1, timeout=45, max_retries=3):
        """观看视频"""
        for attempt in range(max_retries):
            try:
                if not self.driver:
                    self.init_driver()

                logger.info(f"尝试加载视频: {url}")

                try:
                    self.driver.execute_script(f"window.location.href = '{url}';")
                except TimeoutException:
                    logger.warning("页面加载超时，尝试强制停止...")
                    self.driver.execute_script("window.stop();")
                except Exception as e:
                    logger.warning(f"页面导航异常: {str(e)}")
                    continue

                time.sleep(1)  # 等待页面初始加载

                body = self.wait_for_element_safely(By.TAG_NAME, "body", timeout=timeout)
                if not body:
                    raise Exception("页面加载失败")

                video_element = None
                for selector in ['video', '.bilibili-player-video video', '#bilibili-player video']:
                    video_element = self.wait_for_element_safely(By.CSS_SELECTOR, selector, timeout=timeout)
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
                playing_confirmed = False
                check_interval = 0.5

                while time.time() - start_time < duration:
                    try:
                        is_playing = self.driver.execute_script("""
                            const video = document.querySelector('video');
                            return video && !video.paused && video.currentTime > 0;
                        """)

                        if is_playing and not playing_confirmed:
                            playing_confirmed = True
                            logger.info("视频开始播放")

                        if not is_playing:
                            logger.warning("视频未在播放，尝试重新播放...")
                            self.driver.execute_script("""
                                const video = document.querySelector('video');
                                if (video) {
                                    video.play().catch(error => {
                                        console.log('重新播放失败:', error);
                                    });
                                }
                            """)
                    except Exception as e:
                        logger.warning(f"检查视频播放状态失败: {str(e)}")

                    time.sleep(check_interval)

                if playing_confirmed:
                    logger.info(f"成功观看视频 {duration} 秒")
                    return True
                else:
                    raise Exception("无法确认视频是否正常播放")

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
                global_resource_manager.unregister_driver(self)
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

            # 为每个账户准备对应类型的视频列表
            for account in train_accounts:
                if account['group'] == 'state':
                    videos = video_pool['state']
                else:  # non-state组
                    videos = video_pool['non_state']

                videos = video_pool['state'] if account['group'] == 'state' else video_pool['non_state']
                account['training_videos'] = random.sample(videos, videos_per_user)
                account['video_groups'] = [
                    account['training_videos'][i:i + videos_per_group]
                    for i in range(0, len(account['training_videos']), videos_per_group)
                ]
                account['completed_videos_count'] = 0

            video_groups_count = videos_per_user // videos_per_group
            for group_idx in range(video_groups_count):
                logger.info(f"\n=== 开始处理第 {group_idx + 1}/{video_groups_count} 组视频 ===")
                logger.info("当前各账户观看进度:")
                for acc in train_accounts:
                    logger.info(f"账户 {acc['username']}: {acc['completed_videos_count']}/{videos_per_user} 个视频")

                completed_count = 0
                target_count = videos_per_group * len(train_accounts)

                with concurrent.futures.ThreadPoolExecutor(max_workers=len(train_accounts)) as executor:
                    futures = {
                        executor.submit(
                            self.process_video_group,
                            account,
                            group_idx,
                            video_duration,
                            videos_per_user
                        ): account for account in train_accounts
                    }

                    for future in concurrent.futures.as_completed(futures):
                        account = futures[future]
                        try:
                            success_count = future.result()
                            if success_count > 0:
                                completed_count += success_count
                        except Exception as e:
                            logger.error(f"账户 {account['username']} 视频组 {group_idx + 1} 处理失败: {str(e)}")

                # 检查是否所有账户都完成了当前组的视频
                if completed_count < target_count:
                    logger.warning(f"当前组视频未全部完成: {completed_count}/{target_count}")

                # 该组视频看完后休息
                rest_time = (10 + group_idx) * 60
                logger.info(f"\n=== 第 {group_idx + 1} 组视频完成，开始休息 {rest_time / 60} 分钟 ===")
                logger.info("当前各账户观看进度:")
                for acc in train_accounts:
                    logger.info(f"账户 {acc['username']}: {acc['completed_videos_count']}/{videos_per_user} 个视频")
                time.sleep(rest_time)

            # 添加control组账户到返回结果
            control_accounts = [acc for acc in accounts if acc['group'] == 'control']
            return train_accounts + control_accounts

        except Exception as e:
            logger.error(f"预训练组失败: {str(e)}")
            raise

    def process_video_group(self, account, group_idx, video_duration, total_videos):
        """处理单个账户的一组视频"""
        driver = None
        try:
            driver = BilibiliDriver()
            videos = account['video_groups'][group_idx]

            cookies = load_cookies(account['username'])
            if cookies:
                for cookie in cookies:
                    driver.add_cookie(cookie)

            start_count = account['completed_videos_count']
            success_count = 0

            for i, video in enumerate(videos, 1):
                logger.info(f"账户 {account['username']} 开始观看第 {start_count + i}/{total_videos} 个视频: {video}")

                if driver.watch_video(video, duration=video_duration, timeout=EXPERIMENT_CONFIG['TIMEOUT']):
                    with self.training_lock:
                        account['completed_videos_count'] += 1
                        success_count += 1
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

                time.sleep(random.uniform(2, 4))

            self.show_blank_page(driver)
            return success_count

        except Exception as e:
            logger.error(f"处理视频组失败 (用户: {account['username']}, 组: {group_idx}): {str(e)}")
            return 0
        finally:
            if driver:
                driver.close()


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


def init_account_cookies():
    """初始化账户cookies"""
    try:
        driver = BilibiliDriver()
        if driver.get("https://www.bilibili.com"):
            cookies = driver.driver.get_cookies()
            driver.close()
            return cookies
        return None
    except Exception as e:
        logger.error(f"初始化cookies失败: {str(e)}")
        return None


def create_virtual_accounts(num_accounts: int, group_type: str):
    """创建虚拟账户"""
    accounts = []
    for i in range(num_accounts):
        username = generate_username()
        cookies = init_account_cookies()
        if cookies:
            save_cookies(username, cookies)

        account = {
            'username': username,
            'sex': random.choice(['male', 'female']),
            'group': group_type.strip().lower(),
            'watched_videos': [],
            'completed_videos_count': 0
        }
        accounts.append(account)
        logger.info(f"创建用户: {account['username']}, 组别: {account['group']}")
    return accounts


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

            # 验证视频数量是否足够
            if len(self.state_videos) < EXPERIMENT_CONFIG['VIDEOS_PER_USER']:
                raise ValueError(
                    f"state视频数量不足: 需要{EXPERIMENT_CONFIG['VIDEOS_PER_USER']}个，实际只有{len(self.state_videos)}个")
            if len(self.non_state_videos) < EXPERIMENT_CONFIG['VIDEOS_PER_USER']:
                raise ValueError(
                    f"non-state视频数量不足: 需要{EXPERIMENT_CONFIG['VIDEOS_PER_USER']}个，实际只有{len(self.non_state_videos)}个")

            self.video_pool = {
                'state': self.state_videos,
                'non_state': self.non_state_videos
            }

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
        max_workers = max_workers or min(len(accounts), EXPERIMENT_CONFIG['MAX_WORKERS'])

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
            driver = None

            try:
                driver = BilibiliDriver()
                cookies = load_cookies(account['username'])
                if cookies:
                    for cookie in cookies:
                        driver.add_cookie(cookie)

                self.api.set_cookies(cookies)

                # 获取首页视频
                homepage_videos = None
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
                            for related in related_videos[:10]:
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


def run_batch_experiment(batch_number, total_batches, batch_size_per_group, videos_per_user,
                         videos_per_group, video_duration):
    """运行单个批次的实验"""
    try:
        logger.info(f"\n{'=' * 20} 开始批次 {batch_number}/{total_batches} {'=' * 20}")

        batch_manager = BatchManager()
        accounts = batch_manager.create_batch_accounts(batch_size_per_group)

        experiment_manager = ExperimentManager()
        pretrain_manager = PreTrainManager()

        # 执行预训练
        trained_accounts = pretrain_manager.pretrain_group(
            accounts,
            experiment_manager.video_pool,
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

        # 保存观看历史和检查点
        experiment_manager.save_watch_history(trained_accounts)
        experiment_manager.save_checkpoint(trained_accounts, batch_number, results)

        # 记录批次信息
        batch_log_path = f"{DIRS['batch_logs']}/batch_{batch_number}.json"
        batch_info = {
            'batch_number': batch_number,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'accounts_processed': len(trained_accounts),
            'results_collected': len(results),
            'incomplete_accounts': incomplete_accounts,
            'success_rate': (len(trained_accounts) - len(incomplete_accounts)) / len(
                trained_accounts) if trained_accounts else 0,
            'batch_duration': None,  # 需要添加批次执行时长
            'video_success_rate': None,  # 需要添加视频成功率
            'error_summary': None  # 需要添加错误统计
        }
        with open(batch_log_path, 'w', encoding='utf-8') as f:
            json.dump(batch_info, f, ensure_ascii=False, indent=2)

        logger.info(f"\n{'=' * 20} 批次 {batch_number} 完成 {'=' * 20}")
        return results

    except Exception as e:
        logger.error(f"批次 {batch_number} 执行失败: {str(e)}")
        global_experiment_monitor.record_error("BatchError", str(e))
        raise


def main():
    try:
        # 验证参数
        if EXPERIMENT_CONFIG['VIDEOS_PER_USER'] % EXPERIMENT_CONFIG['VIDEOS_PER_GROUP'] != 0:
            raise ValueError(
                f"每用户视频数({EXPERIMENT_CONFIG['VIDEOS_PER_USER']})必须是每组视频数({EXPERIMENT_CONFIG['VIDEOS_PER_GROUP']})的整数倍")

        if EXPERIMENT_CONFIG['TOTAL_USERS_PER_GROUP'] % EXPERIMENT_CONFIG['BATCH_SIZE_PER_GROUP'] != 0:
            raise ValueError(
                f"每组总用户数({EXPERIMENT_CONFIG['TOTAL_USERS_PER_GROUP']})必须是每批次用户数({EXPERIMENT_CONFIG['BATCH_SIZE_PER_GROUP']})的整数倍")

        # 清理并创建目录
        logger.info("开始清理工作目录...")
        DirectoryManager.clean_directories()
        DirectoryManager.validate_directories()

        # 初始化监控器
        global_experiment_monitor.start_experiment()

        total_batches = EXPERIMENT_CONFIG['TOTAL_USERS_PER_GROUP'] // EXPERIMENT_CONFIG['BATCH_SIZE_PER_GROUP']

        logger.info("\n" + "=" * 50)
        logger.info("开始大规模实验...")
        logger.info(f"总用户数: {EXPERIMENT_CONFIG['TOTAL_USERS_PER_GROUP'] * 3}")
        logger.info(f"总批次数: {total_batches}")
        logger.info(f"每批次用户数: {EXPERIMENT_CONFIG['BATCH_SIZE_PER_GROUP'] * 3}")
        logger.info(f"每用户视频数: {EXPERIMENT_CONFIG['VIDEOS_PER_USER']}")
        logger.info(f"每组视频数: {EXPERIMENT_CONFIG['VIDEOS_PER_GROUP']}")
        logger.info(f"视频观看时长: {EXPERIMENT_CONFIG['VIDEO_DURATION']}秒")
        logger.info(
            f"批次间隔: {EXPERIMENT_CONFIG['MIN_BATCH_INTERVAL']}-{EXPERIMENT_CONFIG['MAX_BATCH_INTERVAL']}分钟")
        logger.info("=" * 50)

        batch_manager = BatchManager()
        all_results = []
        batch_success_count = 0

        global_experiment_monitor.update_stats(total_batches=total_batches)

        for batch in range(1, total_batches + 1):
            try:
                batch_results = run_batch_experiment(
                    batch,
                    total_batches,
                    EXPERIMENT_CONFIG['BATCH_SIZE_PER_GROUP'],
                    EXPERIMENT_CONFIG['VIDEOS_PER_USER'],
                    EXPERIMENT_CONFIG['VIDEOS_PER_GROUP'],
                    EXPERIMENT_CONFIG['VIDEO_DURATION']
                )

                if batch_results:
                    all_results.extend(batch_results)
                    batch_success_count += 1
                    global_experiment_monitor.update_stats(successful_batches=1)
                else:
                    global_experiment_monitor.update_stats(failed_batches=1)

                # 更新监控统计
                global_experiment_monitor.update_stats(
                    total_accounts=EXPERIMENT_CONFIG['BATCH_SIZE_PER_GROUP'] * 3,
                    successful_accounts=len(batch_results) if batch_results else 0,
                    failed_accounts=EXPERIMENT_CONFIG['BATCH_SIZE_PER_GROUP'] * 3 - (
                        len(batch_results) if batch_results else 0)
                )

                if batch < total_batches:
                    batch_manager.wait_between_batches(
                        EXPERIMENT_CONFIG['MIN_BATCH_INTERVAL'],
                        EXPERIMENT_CONFIG['MAX_BATCH_INTERVAL']
                    )

            except Exception as e:
                logger.error(f"批次 {batch} 失败: {str(e)}")
                global_experiment_monitor.record_error("BatchError", str(e))
                global_experiment_monitor.update_stats(failed_batches=1)
                continue

        # 清理旧文件
        global_resource_manager.cleanup_old_files()

        # 生成并保存最终报告
        final_report = global_experiment_monitor.generate_report()
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
        global_experiment_monitor.record_error("ExperimentError", str(e))
        raise
    finally:
        # 确保清理所有资源
        global_resource_manager.cleanup_all_drivers()


if __name__ == "__main__":
    try:
        # 设置全局资源管理器和监控器
        global_resource_manager = ResourceManager()
        global_experiment_monitor = ExperimentMonitor()

        # 运行实验
        main()
    except KeyboardInterrupt:
        logger.warning("\n实验被用户中断")
    except Exception as e:
        logger.error(f"\n实验运行失败: {str(e)}")
    finally:
        logger.info("\n实验程序结束")