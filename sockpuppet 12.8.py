# NO

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


# 全局浏览器管理器
class BrowserManager:
    def __init__(self):
        self.browsers = {}  # 存储所有账户的浏览器实例
        self.browser_lock = Lock()

    def create_all_browsers(self, total_accounts_per_group, groups):
        """一次性为指定组创建账户并保持浏览器实例"""
        total_accounts = total_accounts_per_group * len(groups)
        logger.info(f"开始为 {total_accounts} 个账户创建浏览器实例...")

        accounts = []
        for group in groups:
            for _ in range(total_accounts_per_group):
                try:
                    username = generate_username()
                    driver = BilibiliDriver()

                    if not driver.get("https://www.bilibili.com"):
                        raise Exception("无法访问B站首页")

                    cookies = driver.driver.get_cookies()
                    if not cookies:
                        raise Exception("无法获取cookies")

                    account = {
                        'username': username,
                        'sex': random.choice(['male', 'female']),
                        'group': group,
                        'watched_videos': [],
                        'completed_videos_count': 0
                    }

                    save_cookies(username, cookies)

                    with self.browser_lock:
                        self.browsers[username] = driver

                    accounts.append(account)
                    logger.info(f"成功创建账户 {username} ({group}) 的浏览器实例")

                except Exception as e:
                    logger.error(f"创建账户浏览器实例失败: {str(e)}")
                    raise

        return accounts

    def refresh_browser(self, username):
        """刷新指定账户的浏览器页面到B站首页"""
        try:
            with self.browser_lock:
                if username in self.browsers:
                    driver = self.browsers[username]
                    driver.get("https://www.bilibili.com")
                    time.sleep(2)
        except Exception as e:
            logger.error(f"刷新账户 {username} 的浏览器失败: {str(e)}")

    def get_driver(self, username):
        """获取指定账户的浏览器实例"""
        with self.browser_lock:
            return self.browsers.get(username)

    def cleanup_all_browsers(self):
        """清理所有浏览器实例"""
        with self.browser_lock:
            for username, driver in self.browsers.items():
                try:
                    driver.close()
                except Exception as e:
                    logger.error(f"清理浏览器实例失败 ({username}): {str(e)}")
            self.browsers.clear()


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


def setup_logging():
    logger = logging.getLogger('bilibili_experiment')
    logger.setLevel(logging.INFO)

    file_handler = RotatingFileHandler(
        f"{DIRS['logs']}/experiment.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()

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

    def create_batch_accounts(self, batch_size_per_group, browser_manager, groups):  # 添加groups参数
        """创建单个批次的账户"""
        accounts = []
        for group in groups:
            group_accounts = create_virtual_accounts(batch_size_per_group, group, browser_manager)
            accounts.extend(group_accounts)
        return accounts

    def wait_between_batches(self, min_interval, max_interval):
        """批次间等待"""
        wait_time = random.uniform(min_interval * 60, max_interval * 60)
        logger.info(f"等待 {wait_time / 60:.2f} 分钟后开始下一批次...")
        time.sleep(wait_time)

    def process_batch(self, batch_number, total_batches, accounts, exp_manager):
        """处理单个批次"""
        try:
            logger.info(f"开始处理批次 {batch_number}/{total_batches}")
            return exp_manager.process_accounts(accounts)
        except Exception as e:
            logger.error(f"批次 {batch_number} 处理失败: {str(e)}")
            return None



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
            time.sleep(random.uniform(300, 600))
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

    def init_driver(self, max_retries=3):
        """初始化Chrome驱动"""
        for attempt in range(max_retries):
            try:
                self.options = webdriver.ChromeOptions()

                # Chrome 131版本推荐的参数
                self.options.add_argument('--headless=new')  # Chrome 109+的新headless模式
                self.options.add_argument('--no-sandbox')
                self.options.add_argument('--disable-dev-shm-usage')
                self.options.add_argument('--disable-gpu')

                # Chrome 131的性能优化参数
                self.options.add_argument('--enable-features=NetworkService,NetworkServiceInProcess')
                self.options.add_argument('--disable-features=VizDisplayCompositor')
                self.options.add_argument('--force-device-scale-factor=1')

                # 内存管理相关参数
                self.options.add_argument('--disable-dev-shm-usage')
                self.options.add_argument('--disable-browser-side-navigation')
                self.options.add_argument('--dns-prefetch-disable')

                # 设置合理的窗口大小
                self.options.add_argument('--window-size=1920,1080')

                # 日志级别
                self.options.add_argument('--log-level=3')  # 只显示严重错误

                service = webdriver.ChromeService(
                    executable_path='/usr/local/bin/chromedriver',
                    log_output=os.path.join(DIRS['logs'], 'chromedriver.log')  # 添加日志输出
                )

                self.driver = webdriver.Chrome(service=service, options=self.options)

                # 设置更合理的超时时间
                self.driver.set_page_load_timeout(60)  # 增加页面加载超时时间
                self.driver.set_script_timeout(60)
                self.driver.implicitly_wait(20)

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

                if attempt < max_retries - 1:
                    time.sleep(15)  # 增加等待时间到15秒
                else:
                    raise
        return False

    def ensure_session_valid(self):
        """确保浏览器会话有效"""
        try:
            # 尝试执行一个简单的操作来测试会话
            self.driver.current_url
            return True
        except Exception as e:
            logger.warning(f"会话失效，尝试重新初始化: {str(e)}")
            try:
                if self.driver:
                    try:
                        self.driver.quit()
                    except:
                        pass
                return self.init_driver()
            except Exception as e:
                logger.error(f"重新初始化失败: {str(e)}")
                return False

    def get(self, url):
        """改进的URL访问方法"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if not self.ensure_session_valid():
                    continue
                self.driver.get(url)
                return True
            except Exception as e:
                logger.error(f"访问URL失败 {url}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
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

    def watch_video(self, url, duration=1, timeout=120, max_retries=3):
        overall_start_time = time.time()

        for attempt in range(max_retries):
            if time.time() - overall_start_time > timeout * 2:
                logger.warning(f"视频总尝试时间超过限制，放弃: {url}")
                return False

            try:
                if not self.ensure_session_valid():
                    logger.warning("会话无效，重试")
                    continue

                logger.info(f"尝试第 {attempt + 1}/{max_retries} 次加载视频: {url}")
                attempt_start_time = time.time()

                # 加载视频页面
                success = self.get(url)
                if not success:
                    continue

                # 等待页面稳定
                time.sleep(3)

                try:
                    # 检查页面是否正常加载
                    self.driver.execute_script("return document.readyState") == "complete"
                except:
                    logger.warning("页面未完全加载")
                    continue

                # 查找视频元素
                video_element = None
                for selector in ['video', '.bilibili-player-video video', '#bilibili-player video']:
                    try:
                        video_element = self.wait_for_element_safely(By.CSS_SELECTOR, selector, timeout=15)
                        if video_element:
                            break
                    except Exception as e:
                        continue

                if not video_element:
                    continue

                # 监控播放状态
                last_progress_time = time.time()
                last_current_time = 0
                progress_timeout = 30  # 30秒无进度就重试

                while time.time() - attempt_start_time < timeout:
                    if not self.ensure_session_valid():
                        break

                    try:
                        current_time = self.driver.execute_script("""
                            var video = document.querySelector('video');
                            if (!video) return -1;
                            if (video.paused) {
                                video.play();
                                return -1;
                            }
                            return video.currentTime;
                        """)

                        if current_time > last_current_time:
                            last_current_time = current_time
                            last_progress_time = time.time()
                        elif time.time() - last_progress_time > progress_timeout:
                            raise TimeoutException("播放进度停滞")

                        if current_time >= duration:
                            return True

                    except Exception as e:
                        if time.time() - last_progress_time > progress_timeout:
                            break
                        time.sleep(1)
                        continue

                    time.sleep(0.5)

            except Exception as e:
                logger.error(f"观看视频失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if isinstance(e, (TimeoutException, NoSuchElementException)):
                    time.sleep(2)
                else:
                    time.sleep(5)

            finally:
                try:
                    if self.ensure_session_valid():
                        self.get("https://www.bilibili.com")
                except:
                    pass

        return False


class PreTrainManager:
    def __init__(self, browser_manager):
        self.browser_manager = browser_manager
        self.training_lock = Lock()
        self.batch_completion_count = 0
        self.batch_completion_lock = Lock()
        self.batch_event = threading.Event()
        # 新增: 记录每个账户的重试次数
        self.retry_counts = defaultdict(int)
        self.retry_lock = Lock()
        # 新增: 视频观看进度跟踪
        self.progress_tracker = defaultdict(dict)
        self.progress_lock = Lock()

    def pretrain_group(self, accounts, experiment_manager, videos_per_user, videos_per_group, video_duration):
        try:
            # 1. 改进的组别统计
            group_stats = defaultdict(int)
            for acc in accounts:
                group_stats[acc['group']] += 1

            logger.info(f"开始预训练: {dict(group_stats)}")

            # 2. 优化视频分配
            for account in accounts:
                if account['group'] == 'state':
                    available_videos = experiment_manager.state_videos
                elif account['group'] == 'non-state':
                    available_videos = experiment_manager.non_state_videos
                else:
                    continue

                # 确保视频数量足够且不重复
                if len(available_videos) < videos_per_user:
                    raise ValueError(
                        f"视频池不足: {account['group']} 组需要 {videos_per_user} 个视频，但只有 {len(available_videos)} 个")

                # 使用权重随机采样
                account['training_videos'] = self._weighted_sample_videos(
                    available_videos,
                    videos_per_user
                )

                # 分组并添加休息标记
                account['video_groups'] = []
                for i in range(0, len(account['training_videos']), videos_per_group):
                    group = account['training_videos'][i:i + videos_per_group]
                    account['video_groups'].append({
                        'videos': group,
                        'needs_rest': i + videos_per_group < len(account['training_videos'])
                    })

            # 3. 改进的批次处理
            total_groups = len(account['video_groups']) if accounts and 'video_groups' in accounts[0] else 0
            active_accounts = [acc for acc in accounts if acc['group'] != 'control']

            for group_idx in range(total_groups):
                logger.info(f"\n=== 开始处理第 {group_idx + 1}/{total_groups} 组视频 ===")

                self._reset_batch_state()
                batch_results = self._process_video_batch_parallel(
                    active_accounts,
                    group_idx,
                    video_duration,
                    videos_per_group
                )

                # 4. 添加批次完成验证
                if not self._verify_batch_completion(active_accounts, group_idx, videos_per_group):
                    logger.warning(f"批次 {group_idx + 1} 未完全完成，进行补充观看")
                    self._handle_incomplete_views(
                        active_accounts,
                        group_idx,
                        video_duration,
                        videos_per_group
                    )

                # 5. 智能休息时间
                if group_idx < total_groups - 1:
                    rest_time = self._calculate_rest_time(group_idx, total_groups)
                    logger.info(f"\n=== 第 {group_idx + 1} 组视频完成，休息 {rest_time / 60:.1f} 分钟 ===")
                    self._supervised_rest(active_accounts, rest_time)

            return accounts

        except Exception as e:
            logger.error(f"预训练组失败: {str(e)}")
            self._handle_training_failure(accounts, e)
            raise

    def _weighted_sample_videos(self, videos, sample_size):
        """使用加权随机采样选择视频"""
        weights = self._calculate_video_weights(videos)
        return random.choices(videos, weights=weights, k=sample_size)

    def _calculate_video_weights(self, videos):
        """计算视频的采样权重"""
        # 这里可以基于视频特征计算权重
        return [1.0] * len(videos)  # 暂时使用均匀权重

    def _reset_batch_state(self):
        """重置批次状态"""
        self.batch_completion_count = 0
        self.batch_event.clear()
        with self.progress_lock:
            self.progress_tracker.clear()

    def _process_video_batch_parallel(self, accounts, group_idx, video_duration, batch_target):
        """并行处理视频批次"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(accounts)) as executor:
            futures = {
                executor.submit(
                    self._process_account_videos,
                    account,
                    group_idx,
                    video_duration,
                    batch_target
                ): account for account in accounts
            }

            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    account = futures[future]
                    logger.error(f"账户 {account['username']} 处理失败: {str(e)}")
                    self._handle_account_failure(account, e)

            return results

    def _process_account_videos(self, account, group_idx, video_duration, batch_target):
        """处理单个账户的视频观看"""
        try:
            driver = self.browser_manager.get_driver(account['username'])
            if not driver:
                raise Exception(f"找不到账户 {account['username']} 的浏览器实例")

            video_group = account['video_groups'][group_idx]
            videos_watched = 0
            failed_videos = []

            for video in video_group['videos']:
                if videos_watched >= batch_target:
                    break

                success = self._watch_video_with_verification(
                    driver,
                    account,
                    video,
                    video_duration
                )

                if success:
                    videos_watched += 1
                    self._update_progress(account, group_idx, videos_watched)
                else:
                    failed_videos.append(video)

            if video_group['needs_rest']:
                self._take_rest(driver, account)

            return {
                'account': account['username'],
                'success': videos_watched >= batch_target,
                'watched': videos_watched,
                'failed': len(failed_videos)
            }

        except Exception as e:
            logger.error(f"处理账户视频失败 ({account['username']}): {str(e)}")
            return {
                'account': account['username'],
                'success': False,
                'watched': 0,
                'failed': batch_target
            }

    def _watch_video_with_verification(self, driver, account, video, duration):
        """带验证的视频观看"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                if driver.watch_video(video, duration=duration, timeout=60):
                    self._record_successful_view(account, video, duration)
                    return True

                if attempt < max_attempts - 1:
                    logger.warning(f"视频观看失败，重试 ({account['username']}, 尝试 {attempt + 1}/{max_attempts})")
                    time.sleep(5)

            except Exception as e:
                logger.error(f"视频观看出错 ({account['username']}): {str(e)}")
                if attempt < max_attempts - 1:
                    time.sleep(5)

        return False

    def _record_successful_view(self, account, video, duration):
        """记录成功的视频观看"""
        with self.training_lock:
            account['completed_videos_count'] += 1
            watch_record = {
                'url': video,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'duration': duration,
                'video_number': account['completed_videos_count'],
                'success': True
            }
            account['watched_videos'].append(watch_record)

    def _update_progress(self, account, group_idx, videos_watched):
        """更新进度跟踪"""
        with self.progress_lock:
            self.progress_tracker[account['username']] = {
                'group_idx': group_idx,
                'videos_watched': videos_watched
            }

    def _verify_batch_completion(self, accounts, group_idx, target):
        """验证批次完成情况"""
        with self.progress_lock:
            incomplete = [
                acc['username'] for acc in accounts
                if self.progress_tracker.get(acc['username'], {}).get('videos_watched', 0) < target
            ]
            return len(incomplete) == 0

    def _handle_incomplete_views(self, accounts, group_idx, video_duration, target):
        """处理未完成的观看"""
        with self.progress_lock:
            incomplete_accounts = [
                acc for acc in accounts
                if self.progress_tracker.get(acc['username'], {}).get('videos_watched', 0) < target
            ]

            if incomplete_accounts:
                logger.info(f"处理 {len(incomplete_accounts)} 个未完成账户的补充观看")
                self._process_video_batch_parallel(
                    incomplete_accounts,
                    group_idx,
                    video_duration,
                    target
                )

    def _calculate_rest_time(self, group_idx, total_groups):
        """计算智能休息时间"""
        base_rest_time = 600  # 10分钟基础休息时间
        # 根据进度增加休息时间
        progress_factor = (group_idx + 1) / total_groups
        return base_rest_time * (1 + progress_factor * 0.5)

    def _supervised_rest(self, accounts, rest_time):
        """监督休息过程"""
        try:
            # 确保所有账户都回到首页
            for account in accounts:
                driver = self.browser_manager.get_driver(account['username'])
                if driver:
                    driver.get("https://www.bilibili.com")

            # 分段休息，定期检查浏览器状态
            check_interval = 60  # 每60秒检查一次
            remaining_time = rest_time

            while remaining_time > 0:
                sleep_time = min(check_interval, remaining_time)
                time.sleep(sleep_time)
                remaining_time -= sleep_time

                # 检查浏览器状态
                for account in accounts:
                    driver = self.browser_manager.get_driver(account['username'])
                    if driver:
                        driver.ensure_session_valid()

        except Exception as e:
            logger.error(f"休息监督过程出错: {str(e)}")

    def _handle_training_failure(self, accounts, error):
        """处理训练失败情况"""
        logger.error(f"训练失败，尝试保存当前状态: {str(error)}")
        try:
            failure_info = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'error': str(error),
                'accounts_state': {
                    acc['username']: {
                        'completed_videos': acc['completed_videos_count'],
                        'progress': self.progress_tracker.get(acc['username'], {})
                    } for acc in accounts
                }
            }

            with open(f"{DIRS['logs']}/training_failure_{int(time.time())}.json", 'w') as f:
                json.dump(failure_info, f, indent=2)

        except Exception as e:
            logger.error(f"保存失败状态时出错: {str(e)}")

    def _handle_account_failure(self, account, error):
        """处理单个账户失败情况"""
        with self.retry_lock:
            self.retry_counts[account['username']] += 1
            if self.retry_counts[account['username']] >= 3:
                logger.error(f"账户 {account['username']} 失败次数过多，标记为不可用")
                account['status'] = 'disabled'

class ExperimentManager:
    def __init__(self):
        try:
            self.state_videos_csv = f"{BASE_DIR}/state.csv"
            self.non_state_videos_csv = f"{BASE_DIR}/non.csv"

            if not os.path.exists(self.state_videos_csv):
                raise FileNotFoundError(f"找不到state视频文件: {self.state_videos_csv}")
            if not os.path.exists(self.non_state_videos_csv):
                raise FileNotFoundError(f"找不到non-state视频文件: {self.non_state_videos_csv}")

            try:
                self.state_videos = pd.read_csv(self.state_videos_csv)['视频链接'].tolist()
            except UnicodeDecodeError:
                self.state_videos = pd.read_csv(self.state_videos_csv, encoding='gbk')['视频链接'].tolist()

            try:
                self.non_state_videos = pd.read_csv(self.non_state_videos_csv)['视频链接'].tolist()
            except UnicodeDecodeError:
                self.non_state_videos = pd.read_csv(self.non_state_videos_csv, encoding='gbk')['视频链接'].tolist()

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

    def collect_data_for_user(self, account, shared_results):
        """收集单个用户的数据"""
        try:
            driver = self.browser_manager.get_driver(account['username'])
            if not driver:
                raise Exception(f"找不到账户 {account['username']} 的浏览器实例")

            cookies = load_cookies(account['username'])
            self.api.set_cookies(cookies)

            homepage_videos = None
            for attempt in range(3):
                try:
                    homepage_videos = self.api.get_homepage_videos()
                    if homepage_videos:
                        break
                    logger.warning(f"用户 {account['username']} 第 {attempt + 1} 次获取首页视频失败")
                    if attempt < 2:
                        time.sleep(random.uniform(10, 15))
                except Exception as e:
                    if attempt == 2:
                        raise
                    logger.warning(f"获取首页视频出错: {str(e)}，重试...")
                    time.sleep(random.uniform(10, 15))

            if not homepage_videos:
                raise Exception("无法获取首页视频")

            local_results = []
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
            logger.error(f"收集用户 {account['username']} 数据失败: {str(e)}")
            return []


def init_account_cookies():
    """初始化账户cookies"""
    try:
        driver = BilibiliDriver()
        if driver.get("https://www.bilibili.com"):
            time.sleep(2)
            cookies = driver.driver.get_cookies()
            if cookies:
                for cookie in cookies:
                    if not all(k in cookie for k in ['name', 'value', 'domain']):
                        logger.error(f"Cookie格式不完整: {cookie}")
                        return None
                    if not cookie['domain'].endswith('bilibili.com'):
                        logger.error(f"无效的cookie domain: {cookie['domain']}")
                        return None
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


def create_virtual_accounts(num_accounts: int, group_type: str, browser_manager):
    """创建虚拟账户"""
    accounts = []
    attempt = 0

    while len(accounts) < num_accounts:
        attempt += 1
        try:
            username = generate_username()
            driver = BilibiliDriver()

            if not driver.get("https://www.bilibili.com"):
                logger.warning(f"第 {attempt} 次尝试: 访问B站失败，将重试")
                continue

            cookies = driver.driver.get_cookies()
            if not cookies:
                logger.warning(f"第 {attempt} 次尝试: 获取cookies失败，将重试")
                continue

            save_cookies(username, cookies)
            loaded_cookies = load_cookies(username)

            if not loaded_cookies:
                logger.warning(f"第 {attempt} 次尝试: cookies保存或加载失败，将重试")
                continue

            # 保存浏览器实例
            browser_manager.browsers[username] = driver

            account = {
                'username': username,
                'sex': random.choice(['male', 'female']),
                'group': group_type.strip().lower(),
                'watched_videos': [],
                'completed_videos_count': 0
            }
            accounts.append(account)
            logger.info(f"第 {attempt} 次尝试: 成功创建用户 {username}, 组别: {group_type}, 进度: {len(accounts)}/{num_accounts}")

        except Exception as e:
            logger.error(f"第 {attempt} 次尝试: 创建账户失败: {str(e)}")
            continue

    logger.info(f"完成 {group_type} 组账户创建，共尝试 {attempt} 次，成功创建 {len(accounts)} 个账户")
    return accounts


def run_batch_experiment(round_number, total_rounds, concurrent_users_per_group, videos_per_user,
                        videos_per_group, video_duration):
    try:
        logger.info(f"\n{'=' * 20} 开始第 {round_number}/{total_rounds} 轮 {'=' * 20}")

        # 创建批次管理器和浏览器管理器
        batch_manager = BatchManager()
        browser_manager = BrowserManager()

        try:
            # 第一阶段：只创建state和non-state组账户进行预训练
            # 每组创建concurrent_users_per_group个账户
            training_accounts = browser_manager.create_all_browsers(
                concurrent_users_per_group,  # 每组10个
                groups=['state', 'non-state']
            )

            # 初始化实验管理器
            experiment_manager = ExperimentManager()
            experiment_manager.browser_manager = browser_manager
            pretrain_manager = PreTrainManager(browser_manager)

            # 执行预训练
            logger.info(f"第 {round_number} 轮：开始预训练 state和non-state组账户")
            trained_accounts = pretrain_manager.pretrain_group(
                training_accounts,
                experiment_manager,
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
                        f"第 {round_number} 轮：账户 {account['username']} 预训练未完成: "
                        f"{account['completed_videos_count']}/{videos_per_user}"
                    )

            # 预训练完成后的休息时间
            rest_time = (10 + videos_per_user // videos_per_group) * 60
            logger.info(f"\n=== 第 {round_number} 轮预训练完成，休息 {rest_time/60} 分钟 ===")
            time.sleep(rest_time)

            # 第二阶段：创建control组账户
            logger.info(f"\n=== 第 {round_number} 轮：开始创建control组账户 ===")
            control_accounts = browser_manager.create_all_browsers(
                concurrent_users_per_group,  # 10个
                groups=['control']
            )

            # 合并当前轮次的所有账户
            all_accounts = trained_accounts + control_accounts
            logger.info(f"\n=== 第 {round_number} 轮：开始三组并行实验，共 {len(all_accounts)} 个账户 ===")

            # 收集数据
            results = experiment_manager.collect_data_parallel(all_accounts)

            # 保存当前轮次的观看历史和检查点
            experiment_manager.save_watch_history(all_accounts)
            experiment_manager.save_checkpoint(all_accounts, round_number, results)

            # 记录轮次完成信息
            batch_log_path = f"{DIRS['batch_logs']}/round_{round_number}.json"
            batch_info = {
                'round_number': round_number,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'accounts_processed': len(all_accounts),
                'results_collected': len(results),
                'incomplete_accounts': incomplete_accounts,
                'success_rate': (len(trained_accounts) - len(incomplete_accounts)) / len(
                    trained_accounts) if trained_accounts else 0
            }
            with open(batch_log_path, 'w', encoding='utf-8') as f:
                json.dump(batch_info, f, ensure_ascii=False, indent=2)

            logger.info(f"\n{'=' * 20} 第 {round_number} 轮完成 {'=' * 20}")
            return results

        finally:
            # 清理浏览器实例
            browser_manager.cleanup_all_browsers()

    except Exception as e:
        logger.error(f"第 {round_number} 轮执行失败: {str(e)}")
        raise


def main():
    # 实验参数设置
    TOTAL_USERS_PER_GROUP = 30  # 每组总用户数改为30
    CONCURRENT_USERS_PER_GROUP = 10  # 每组并发用户数保持10
    VIDEOS_PER_USER = 100  # 每个用户要观看的视频总数
    VIDEOS_PER_GROUP = 10  # 每组视频数
    VIDEO_DURATION = 30  # 每个视频观看时长（秒）
    MIN_BATCH_INTERVAL = 28  # 批次间最小间隔（分钟）
    MAX_BATCH_INTERVAL = 33  # 批次间最大间隔（分钟）

    try:
        # 验证参数
        if VIDEOS_PER_USER % VIDEOS_PER_GROUP != 0:
            raise ValueError(f"每用户视频数({VIDEOS_PER_USER})必须是每组视频数({VIDEOS_PER_GROUP})的整数倍")
        if TOTAL_USERS_PER_GROUP % CONCURRENT_USERS_PER_GROUP != 0:
            raise ValueError(
                f"每组总用户数({TOTAL_USERS_PER_GROUP})必须是并发用户数({CONCURRENT_USERS_PER_GROUP})的整数倍")

        # 清理目录
        DirectoryManager.clean_directories()
        DirectoryManager.ensure_directories()

        # 初始化管理器
        resource_manager = ResourceManager()
        experiment_monitor = ExperimentMonitor()
        experiment_monitor.start_experiment()

        # 计算需要执行的总轮次
        total_rounds = TOTAL_USERS_PER_GROUP // CONCURRENT_USERS_PER_GROUP

        logger.info("\n" + "=" * 50)
        logger.info("开始大规模实验...")
        logger.info(f"每组总用户数: {TOTAL_USERS_PER_GROUP}")
        logger.info(f"每组并发用户数: {CONCURRENT_USERS_PER_GROUP}")
        logger.info(f"总轮次数: {total_rounds}")
        logger.info(f"每轮并发总人数: {CONCURRENT_USERS_PER_GROUP * 3}")
        logger.info(f"每用户视频数: {VIDEOS_PER_USER}")
        logger.info(f"每组视频数: {VIDEOS_PER_GROUP}")
        logger.info(f"视频观看时长: {VIDEO_DURATION}秒")
        logger.info(f"轮次间隔: {MIN_BATCH_INTERVAL}-{MAX_BATCH_INTERVAL}分钟")
        logger.info("=" * 50)

        batch_manager = BatchManager()
        all_results = []
        round_success_count = 0

        for round_num in range(1, total_rounds + 1):
            try:
                logger.info(f"\n{'=' * 20} 开始第 {round_num}/{total_rounds} 轮 {'=' * 20}")

                # 每轮创建30个并发账号(每组10个)
                round_results = run_batch_experiment(
                    round_num,
                    total_rounds,
                    CONCURRENT_USERS_PER_GROUP,  # 使用并发用户数而不是总用户数
                    VIDEOS_PER_USER,
                    VIDEOS_PER_GROUP,
                    VIDEO_DURATION
                )

                if round_results:
                    all_results.extend(round_results)
                    round_success_count += 1

                experiment_monitor.update_stats(
                    total_accounts=CONCURRENT_USERS_PER_GROUP * 3,
                    successful_accounts=len(round_results) if round_results else 0,
                    failed_accounts=CONCURRENT_USERS_PER_GROUP * 3 - (len(round_results) if round_results else 0)
                )

                # 在轮次之间添加休息时间
                if round_num < total_rounds:
                    rest_time = random.uniform(MIN_BATCH_INTERVAL * 60, MAX_BATCH_INTERVAL * 60)
                    logger.info(f"\n=== 第 {round_num} 轮完成，休息 {rest_time / 60:.1f} 分钟后开始下一轮 ===")
                    time.sleep(rest_time)

            except Exception as e:
                logger.error(f"第 {round_num} 轮执行失败: {str(e)}")
                experiment_monitor.record_error("RoundError", str(e))
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
        logger.info(f"总执行轮次: {total_rounds}")
        logger.info(f"成功轮次数: {round_success_count}")
        logger.info(f"总收集数据条数: {len(all_results)}")
        logger.info(f"最终报告已保存至: {report_path}")
        logger.info("=" * 50 + "\n")

    except Exception as e:
        logger.error(f"实验过程中出错: {str(e)}")
        experiment_monitor.record_error("ExperimentError", str(e))
        raise
    finally:
        logger.info("实验结束，保持所有浏览器窗口运行")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n实验被用户中断")
    except Exception as e:
        logger.error(f"\n实验运行失败: {str(e)}")
    finally:
        logger.info("\n实验程序结束")