# First, ensure these packages are installed using pip:
# pip install psutil

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
import gc  # Add explicit import for garbage collector
import psutil  # Add explicit import for process utilities
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
        self.browsers = {}
        self.browser_lock = Lock()

    def cleanup_all_browsers(self):
        """改进的浏览器清理方法"""
        cleanup_count = 0
        error_count = 0

        with self.browser_lock:
            if not self.browsers:
                logger.info("没有需要清理的浏览器实例")
                return

            # 获取所有需要清理的浏览器实例
            browsers_to_cleanup = list(self.browsers.items())
            logger.info(f"开始清理 {len(browsers_to_cleanup)} 个浏览器实例...")

            for username, driver in browsers_to_cleanup:
                try:
                    # 先导航到空白页面，减少内存占用
                    if driver and driver.driver:
                        try:
                            driver.driver.get("about:blank")
                        except:
                            pass

                    # 清理每个标签页
                    if driver and driver.driver:
                        try:
                            handles = driver.driver.window_handles[:]
                            for handle in handles:
                                try:
                                    driver.driver.switch_to.window(handle)
                                    driver.driver.close()
                                except Exception as e:
                                    logger.error(f"关闭标签页失败 ({username}): {str(e)}")
                        except Exception as e:
                            logger.error(f"处理标签页时出错 ({username}): {str(e)}")

                    # 退出浏览器
                    if driver and driver.driver:
                        try:
                            driver.driver.quit()
                            cleanup_count += 1
                            logger.info(f"成功清理浏览器实例: {username}")
                        except Exception as e:
                            logger.error(f"退出浏览器失败 ({username}): {str(e)}")
                            error_count += 1

                except Exception as e:
                    logger.error(f"清理浏览器时出错 ({username}): {str(e)}")
                    error_count += 1
                finally:
                    # 确保从字典中移除
                    self.browsers.pop(username, None)

            # 清理完成后强制回收
            try:
                gc.collect()
                time.sleep(2)  # 给系统一点时间来回收资源
            except Exception as e:
                logger.error(f"垃圾回收时出错: {str(e)}")

            logger.info(f"浏览器清理完成: 成功 {cleanup_count} 个, 失败 {error_count} 个")

            # 确保字典为空
            self.browsers.clear()

    def create_all_browsers(self, total_accounts_per_group, groups):
        """改进的浏览器创建方法"""
        total_accounts = total_accounts_per_group * len(groups)
        logger.info(f"开始为 {total_accounts} 个账户创建浏览器实例...")

        accounts = []

        # 确保之前的浏览器都已清理
        self.cleanup_all_browsers()

        max_browsers_allowed = total_accounts_per_group * len(groups)
        current_browser_count = 0

        for group in groups:
            for i in range(total_accounts_per_group):
                if current_browser_count >= max_browsers_allowed:
                    logger.warning(f"已达到最大浏览器数量限制: {max_browsers_allowed}")
                    return accounts

                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        username = generate_username()

                        # 检查当前活动的浏览器数量
                        if len(self.browsers) >= max_browsers_allowed:
                            logger.warning("浏览器实例数量超过限制，清理现有实例...")
                            self.cleanup_all_browsers()
                            gc.collect()
                            time.sleep(5)

                        # 添加内存检查
                        process = psutil.Process()
                        memory_percent = process.memory_percent()

                        if memory_percent > 80:  # 如果内存使用超过80%
                            logger.warning(f"内存使用率过高 ({memory_percent:.1f}%)，等待回收...")
                            self.cleanup_all_browsers()
                            gc.collect()
                            time.sleep(30)  # 等待30秒让系统回收资源

                        driver = BilibiliDriver()

                        if not driver.get("https://www.bilibili.com"):
                            raise Exception("无法访问B站首页")

                        cookies = driver.driver.get_cookies()
                        save_cookies(username, cookies)

                        account = {
                            'username': username,
                            'sex': random.choice(['male', 'female']),
                            'group': group,
                            'watched_videos': [],
                            'completed_videos_count': 0
                        }

                        with self.browser_lock:
                            self.browsers[username] = driver

                        accounts.append(account)
                        current_browser_count += 1
                        logger.info(f"成功创建账户 {username} ({group}) [{i + 1}/{total_accounts_per_group}]")
                        break

                    except Exception as e:
                        logger.error(f"创建账户尝试 {attempt + 1}/{max_retries} 失败: {str(e)}")
                        if attempt == max_retries - 1:
                            raise
                        time.sleep(random.uniform(5, 10))

        return accounts


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

    def get_homepage_videos(self):
        """获取首页推荐视频"""
        for attempt in range(3):
            try:
                self._wait_between_requests()  # 添加请求间隔

                endpoint = f"{self.base_url}/x/web-interface/index/top/feed/rcmd"
                params = {
                    'ps': 30,
                    'fresh_type': 3,
                    'fresh_idx': random.randint(1, 100),
                    'fresh_idx_1h': random.randint(1, 100),
                    'feed_version': 'V8'
                }

                headers = {
                    'User-Agent': self._get_random_ua(),
                    'Referer': 'https://www.bilibili.com',
                    'Origin': 'https://www.bilibili.com'
                }

                response = self.session.get(
                    endpoint,
                    params=params,
                    headers=headers,
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()
                    if data.get('code') == 0:
                        self.error_count = 0  # 成功后重置错误计数
                        return data['data'].get('item', [])

                self._handle_request_error("API请求失败")

            except Exception as e:
                logger.error(f"获取首页视频失败 (尝试 {attempt + 1}/3): {str(e)}")
                self._handle_request_error(str(e))

            if attempt < 2:
                time.sleep(random.uniform(5, 10))

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

    def request_with_retry(self, endpoint, params=None, max_retries=3):
        """带重试机制的请求方法"""
        retry_count = 0
        while retry_count < max_retries:
            try:
                # 限制请求频率
                with self.request_lock:
                    current_time = time.time()
                    if current_time - self.last_request_time < 1:  # 至少间隔1秒
                        time.sleep(1 - (current_time - self.last_request_time))

                    headers = {
                        'User-Agent': self._get_random_ua(),
                        'Referer': 'https://www.bilibili.com'
                    }

                    response = self.session.get(endpoint, params=params, headers=headers, timeout=30)
                    self.last_request_time = time.time()

                    if response.status_code == 200:
                        try:
                            data = response.json()
                            if data.get('code') == 0:
                                self.error_count = 0  # 成功后重置错误计数
                                return data
                            else:
                                logger.warning(f"B站API返回错误: {data.get('message', '未知错误')}")
                        except ValueError:
                            logger.error("解析JSON响应失败")

                # 请求失败，增加错误计数
                self.error_count += 1

                # 检查是否需要长时间休息
                if self._should_reset_error_count():
                    retry_count = 0  # 重置重试次数
                    continue

            except Exception as e:
                logger.error(f"请求失败 (尝试 {retry_count + 1}/{max_retries}): {str(e)}")

            # 增加重试延迟
            retry_delay = min(2 ** retry_count * 5, 60)  # 指数退避，最长60秒
            time.sleep(retry_delay)
            retry_count += 1

        return None

    def _wait_between_requests(self):
        """请求之间的等待"""
        with self.request_lock:
            current_time = time.time()
            # 确保请求间隔至少2秒
            if current_time - self.last_request_time < 2:
                sleep_time = 2 - (current_time - self.last_request_time)
                time.sleep(sleep_time)
            self.last_request_time = time.time()

    def _handle_request_error(self, error_msg):
        """处理请求错误"""
        self.error_count += 1
        if self.error_count >= self.MAX_ERROR_COUNT:
            sleep_time = random.uniform(60, 120)  # 错误过多时休息1-2分钟
            logger.warning(f"连续错误次数过多，休息 {sleep_time:.1f} 秒")
            time.sleep(sleep_time)
            self.error_count = 0

class BilibiliDriver:
    def __init__(self, profile_id=None):
        self.driver = None
        self.profile_id = profile_id
        self.profile_dir = f"{BASE_DIR}/profiles/{profile_id}" if profile_id else None
        success = self.init_driver()
        if not success:
            raise Exception("Failed to initialize driver")

    def init_driver(self, max_retries=3):
        for attempt in range(max_retries):
            try:
                options = webdriver.ChromeOptions()

                # 添加更多内存优化参数
                options.add_argument('--headless=new')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                options.add_argument('--disable-gpu')
                options.add_argument('--disable-software-rasterizer')
                options.add_argument('--disable-extensions')
                options.add_argument('--disable-logging')
                options.add_argument('--disable-notifications')
                options.add_argument('--disable-default-apps')
                options.add_argument('--disable-background-networking')
                options.add_argument('--disable-sync')
                options.add_argument('--disable-translate')
                options.add_argument('--disable-webgl')
                options.add_argument('--disable-popup-blocking')
                options.add_argument('--disable-plugins')
                options.add_argument('--disable-plugins-discovery')
                options.add_argument('--mute-audio')
                options.add_argument('--disable-javascript-harmony-shipping')

                # 限制内存使用
                options.add_argument('--memory-pressure-off')
                options.add_argument('--js-flags="--max-old-space-size=512"')

                # 添加进程限制
                options.add_argument('--single-process')
                options.add_argument('--disable-renderer-backgrounding')

                if self.profile_dir:
                    os.makedirs(self.profile_dir, exist_ok=True)
                    options.add_argument(f'user-data-dir={self.profile_dir}')

                service = webdriver.ChromeService(
                    executable_path='/usr/local/bin/chromedriver',
                    log_output=os.path.join(DIRS['logs'], 'chromedriver.log')
                )

                self.driver = webdriver.Chrome(service=service, options=options)
                self.driver.set_page_load_timeout(30)
                self.driver.set_script_timeout(30)
                self.driver.implicitly_wait(10)

                return True

            except Exception as e:
                logger.error(f"Driver初始化失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if self.driver:
                    try:
                        self.driver.quit()
                    except:
                        pass
                    self.driver = None

                if attempt < max_retries - 1:
                    time.sleep(random.uniform(5, 10))
                else:
                    raise

        return False

    def ensure_session_valid(self):
        """确保浏览器会话有效"""
        try:
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
        """访问URL的改进方法"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if not self.ensure_session_valid():
                    if attempt == max_retries - 1:
                        return False
                    time.sleep(5)
                    continue

                self.driver.get(url)
                return True

            except Exception as e:
                logger.error(f"访问URL失败 {url}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
        return False

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

    def watch_video(self, url, duration=30, timeout=60):
        """改进的视频观看方法"""
        start_time = time.time()
        max_retries = 3

        for attempt in range(max_retries):
            try:
                # 1. 加载视频页面
                self.driver.get(url)
                time.sleep(3)  # 等待页面开始加载

                # 2. 等待视频播放器加载
                video_element = None
                for selector in ['video', '.bilibili-player-video video', '#bilibili-player video']:
                    try:
                        video_element = WebDriverWait(self.driver, 20).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                        if video_element:
                            break
                    except:
                        continue

                if not video_element:
                    logger.warning(f"尝试 {attempt + 1}: 无法找到视频元素")
                    if attempt < max_retries - 1:
                        time.sleep(random.uniform(3, 5))
                    continue

                # 3. 确保视频开始播放
                play_success = self.driver.execute_script("""
                    var video = document.querySelector('video');
                    if(video) {
                        video.play();
                        // 尝试跳过广告
                        if(document.querySelector('.bilibili-player-video-btn-jump')) {
                            document.querySelector('.bilibili-player-video-btn-jump').click();
                        }
                        return true;
                    }
                    return false;
                """)

                if not play_success:
                    logger.warning(f"尝试 {attempt + 1}: 无法播放视频")
                    if attempt < max_retries - 1:
                        time.sleep(random.uniform(3, 5))
                    continue

                # 4. 监控播放进度
                last_time = 0
                stall_start = None
                progress_check_start = time.time()

                while time.time() - start_time < timeout:
                    try:
                        current_time = self.driver.execute_script(
                            "return document.querySelector('video').currentTime"
                        )

                        # 检查播放是否停滞
                        if current_time == last_time:
                            if stall_start is None:
                                stall_start = time.time()
                            elif time.time() - stall_start > 10:  # 停滞超过10秒
                                raise Exception("视频播放停滞")
                        else:
                            stall_start = None
                            last_time = current_time

                        # 检查是否达到观看时长
                        if current_time >= duration:
                            logger.info(f"成功观看视频 {url} {duration}秒")
                            return True

                        # 超时检查
                        if time.time() - progress_check_start > timeout:
                            raise Exception("观看超时")

                    except Exception as e:
                        logger.warning(f"监控播放进度时出错: {str(e)}")

                    time.sleep(1)

            except Exception as e:
                logger.error(f"观看视频失败 ({url}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(3, 5))
                continue

            finally:
                # 尝试暂停视频并返回首页
                try:
                    self.driver.execute_script("document.querySelector('video').pause()")
                    self.get("https://www.bilibili.com")
                except:
                    pass

        return False

    def close(self):
        """关闭浏览器实例"""
        try:
            if self.driver:
                self.driver.quit()
        except Exception as e:
            logger.error(f"关闭浏览器失败: {str(e)}")


class PreTrainManager:
    def __init__(self, browser_manager):
        self.browser_manager = browser_manager
        self.training_lock = Lock()
        self.batch_completion_count = 0
        self.batch_completion_lock = Lock()
        self.batch_event = threading.Event()
        self.retry_counts = defaultdict(int)
        self.retry_lock = Lock()
        self.progress_tracker = defaultdict(dict)
        self.progress_lock = Lock()

    def pretrain_group(self, accounts, experiment_manager, videos_per_user, videos_per_group, video_duration):
        """
        预训练一组账户
        Args:
            accounts: 需要训练的账户列表
            experiment_manager: 实验管理器
            videos_per_user: 每个用户需要观看的总视频数(100)
            videos_per_group: 每组视频数(10)
            video_duration: 每个视频观看时长
        """
        try:
            # 记录每个组别的账户数量
            group_stats = defaultdict(int)
            for acc in accounts:
                group_stats[acc['group']] += 1
            logger.info(f"开始预训练: {dict(group_stats)}")

            # 为每个账户分配视频
            for account in accounts:
                if account['group'] == 'state':
                    available_videos = experiment_manager.state_videos
                elif account['group'] == 'non-state':
                    available_videos = experiment_manager.non_state_videos
                else:
                    continue

                # 确保视频池足够大
                if len(available_videos) < videos_per_user:
                    raise ValueError(
                        f"视频池不足: {account['group']} 组需要 {videos_per_user} 个视频，但只有 {len(available_videos)} 个")

                # 随机选择视频
                account['training_videos'] = random.sample(available_videos, videos_per_user)

            # 开始训练过程
            for account in accounts:
                if account['group'] != 'control':
                    logger.info(f"开始训练账户 {account['username']} ({account['group']})")
                    driver = self.browser_manager.get_driver(account['username'])

                    if not driver:
                        logger.error(f"无法获取账户 {account['username']} 的浏览器实例")
                        continue

                    videos_watched = 0

                    # 分批观看视频(每组10个)
                    for batch_start in range(0, videos_per_user, videos_per_group):
                        batch_end = min(batch_start + videos_per_group, videos_per_user)
                        current_batch = account['training_videos'][batch_start:batch_end]

                        logger.info(
                            f"账户 {account['username']} 开始观看第 {batch_start // videos_per_group + 1} 组视频")

                        # 观看当前批次的视频
                        for video_url in current_batch:
                            try:
                                # 确保会话有效
                                if not driver.driver or not self._check_session(driver):
                                    logger.warning(f"账户 {account['username']} 会话失效，尝试重新初始化")
                                    if not self._reinitialize_session(account):
                                        raise Exception("无法恢复会话")
                                    driver = self.browser_manager.get_driver(account['username'])

                                # 观看视频
                                success = driver.watch_video(video_url, duration=video_duration)
                                if success:
                                    videos_watched += 1
                                    account['completed_videos_count'] += 1

                                    # 记录观看记录
                                    watch_record = {
                                        'url': video_url,
                                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                        'duration': video_duration,
                                        'video_number': account['completed_videos_count'],
                                        'success': True
                                    }
                                    account['watched_videos'].append(watch_record)
                                    logger.info(
                                        f"账户 {account['username']} 成功观看第 {account['completed_videos_count']} 个视频")
                                else:
                                    logger.warning(f"账户 {account['username']} 观看视频 {video_url} 失败")

                            except Exception as e:
                                logger.error(f"观看视频时出错 ({account['username']}): {str(e)}")
                                continue

                        # 每组视频观看完后休息(除了最后一组)
                        if batch_end < videos_per_user:
                            try:
                                # 回到首页休息
                                driver.driver.get("https://www.bilibili.com")
                                rest_time = random.uniform(300, 600)  # 5-10分钟随机休息
                                logger.info(f"账户 {account['username']} 完成第 {batch_end} 个视频，"
                                            f"休息 {rest_time / 60:.1f} 分钟")
                                time.sleep(rest_time)
                            except Exception as e:
                                logger.error(f"休息过程出错 ({account['username']}): {str(e)}")

            return accounts

        except Exception as e:
            logger.error(f"预训练过程失败: {str(e)}")
            self._handle_training_failure(accounts, e)
            raise

    def _check_session(self, driver):
        """检查浏览器会话是否有效"""
        try:
            driver.driver.current_url
            return True
        except:
            return False

    def _reinitialize_session(self, account):
        """重新初始化会话"""
        try:
            driver = BilibiliDriver(profile_id=account['username'])
            with self.browser_manager.browser_lock:
                self.browser_manager.browsers[account['username']] = driver
            return True
        except:
            return False

    def _handle_training_failure(self, accounts, error):
        """处理训练失败的情况"""
        logger.error(f"训练失败，保存当前状态: {str(error)}")
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
        # 限制最大并发数为10
        max_workers = min(len(accounts), 10)
        active_tasks = 0

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}

                # 为每个账户创建任务
                for account in accounts:
                    try:
                        future = executor.submit(self.collect_data_for_user, account, shared_results)
                        futures[future] = account
                        active_tasks += 1
                        logger.info(f"启动数据收集任务: {account['username']} (活动任务: {active_tasks})")
                    except Exception as e:
                        logger.error(f"创建数据收集任务失败 ({account['username']}): {str(e)}")

                # 处理完成的任务
                for future in concurrent.futures.as_completed(futures):
                    account = futures[future]
                    try:
                        future.result(timeout=300)  # 5分钟超时
                        active_tasks -= 1
                        logger.info(f"完成数据收集任务: {account['username']} (剩余任务: {active_tasks})")
                    except concurrent.futures.TimeoutError:
                        logger.error(f"数据收集任务超时 ({account['username']})")
                    except Exception as e:
                        logger.error(f"数据收集任务失败 ({account['username']}): {str(e)}")
                    finally:
                        active_tasks = max(0, active_tasks - 1)

        except Exception as e:
            logger.error(f"并行数据收集过程出错: {str(e)}")
        finally:
            # 确保清理资源
            gc.collect()

        return shared_results

    def save_watch_history(self, accounts):
        """保存观看历史"""
        try:
            watch_history_path = f"{DIRS['results']}/watch_history_{int(time.time())}.csv"
            temp_path = f"{watch_history_path}.tmp"

            with open(temp_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['username', 'group', 'video_url', 'watch_timestamp',
                                 'duration', 'video_number', 'success'])

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

            # 完成写入后再重命名文件
            os.replace(temp_path, watch_history_path)
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


def run_batch_experiment(round_number, total_rounds, concurrent_users_per_group, videos_per_user, videos_per_group,
                         video_duration):
    """
    运行批次实验
    concurrent_users_per_group: 每组并发用户数（如10）
    """
    try:
        logger.info(f"\n{'=' * 20} 开始第 {round_number}/{total_rounds} 轮 {'=' * 20}")

        batch_manager = BatchManager()
        browser_manager = BrowserManager()
        experiment_manager = ExperimentManager()
        experiment_manager.browser_manager = browser_manager
        pretrain_manager = PreTrainManager(browser_manager)

        # 创建当前批次的训练账户部分
        max_retries = 3
        training_accounts = None

        for attempt in range(max_retries):
            try:
                # 先确保清理之前的实例
                browser_manager.cleanup_all_browsers()
                gc.collect()  # 添加强制垃圾回收
                time.sleep(5)  # 添加短暂等待，确保资源释放

                training_accounts = browser_manager.create_all_browsers(
                    concurrent_users_per_group,
                    groups=['state', 'non-state']
                )

                if training_accounts:
                    break

            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"创建训练账户失败，已重试{max_retries}次: {str(e)}")
                logger.error(f"创建训练账户失败，尝试 {attempt + 1}/{max_retries}: {str(e)}")
                time.sleep(random.uniform(30, 60))

        if not training_accounts:
            raise Exception("无法创建训练账户")


        # 训练阶段
        trained_accounts = pretrain_manager.pretrain_group(
            training_accounts,
            experiment_manager,
            videos_per_user,
            videos_per_group,
            video_duration
        )

        # 训练后休息
        rest_time = random.uniform(300, 600)  # 5-10分钟
        logger.info(f"\n=== 训练阶段完成，休息 {rest_time / 60:.1f} 分钟 ===")
        time.sleep(rest_time)

        # 创建control组账户（10个）
        max_retries = 3
        control_accounts = None

        for attempt in range(max_retries):
            try:
                # 先清理一下资源
                browser_manager.cleanup_all_browsers()
                gc.collect()
                time.sleep(5)  # 等待资源释放

                control_accounts = browser_manager.create_all_browsers(
                    concurrent_users_per_group,
                    groups=['control']
                )

                if control_accounts:
                    break

            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"创建对照组账户失败，已重试{max_retries}次: {str(e)}")
                logger.error(f"创建对照组账户失败，尝试 {attempt + 1}/{max_retries}: {str(e)}")
                time.sleep(random.uniform(30, 60))

        if not control_accounts:
            raise Exception("无法创建对照组账户")

        # 合并所有账号准备收集数据（总共30个）
        all_accounts = trained_accounts + control_accounts

        # 收集数据
        logger.info(f"开始收集数据，共 {len(all_accounts)} 个账号")
        results = experiment_manager.collect_data_parallel(all_accounts)

        # 保存数据
        experiment_manager.save_watch_history(all_accounts)
        experiment_manager.save_checkpoint(all_accounts, round_number, results)

        # 清理当前批次的所有cookies
        try:
            logger.info(f"开始清理当前批次的cookies，涉及 {len(all_accounts)} 个账户...")
            cookies_cleaned = 0
            for account in all_accounts:
                cookie_path = os.path.join(DIRS['cookies'], f"{account['username']}.json")
                if os.path.exists(cookie_path):
                    try:
                        os.remove(cookie_path)
                        cookies_cleaned += 1
                    except Exception as e:
                        logger.error(f"清理 {account['username']} 的cookies失败: {str(e)}")
            logger.info(f"成功清理了 {cookies_cleaned} 个cookies文件")
        except Exception as e:
            logger.error(f"清理批次cookies时发生错误: {str(e)}")

        # 清理当前批次的所有浏览器实例
        browser_manager.cleanup_all_browsers()

        # 本轮休息，准备下一轮
        if round_number < total_rounds:
            next_round_rest = random.uniform(900, 1200)  # 15-20分钟
            logger.info(f"\n=== 第 {round_number} 轮完成，休息 {next_round_rest / 60:.1f} 分钟后开始下一轮 ===")
            time.sleep(next_round_rest)

        return results

    except Exception as e:
        logger.error(f"第 {round_number} 轮执行失败: {str(e)}")
        raise


def main():
    # 实验参数设置
    TOTAL_USERS_PER_GROUP = 5  # 每组总用户数
    CONCURRENT_USERS_PER_GROUP = 5  # 每组并发用户数保持5
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
        browser_manager = BrowserManager()

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
                # 检查和清理资源
                memory_before = psutil.Process().memory_percent()
                logger.info(f"当前轮次开始前内存使用率: {memory_before:.1f}%")

                # 强制清理
                browser_manager.cleanup_all_browsers()
                gc.collect()

                # 检查清理效果
                memory_after = psutil.Process().memory_percent()
                logger.info(f"清理后内存使用率: {memory_after:.1f}%")

                if memory_after > 80:
                    logger.warning(f"内存使用率仍然过高，等待系统回收...")
                    time.sleep(180)  # 等待3分钟
                    gc.collect()

                logger.info(f"\n{'=' * 20} 开始第 {round_num}/{total_rounds} 轮 {'=' * 20}")

                # 记录本轮开始时间
                round_start_time = time.time()

                # 运行本轮实验
                try:
                    round_results = run_batch_experiment(
                        round_num,
                        total_rounds,
                        CONCURRENT_USERS_PER_GROUP,
                        VIDEOS_PER_USER,
                        VIDEOS_PER_GROUP,
                        VIDEO_DURATION
                    )
                except Exception as e:
                    logger.error(f"第 {round_num} 轮执行失败: {str(e)}")
                    # 发生错误时也要确保清理
                    browser_manager.cleanup_all_browsers()
                    gc.collect()
                    # 短暂休息后继续下一轮
                    time.sleep(300)  # 5分钟
                    continue

                # 更新统计
                if round_results:
                    all_results.extend(round_results)
                    round_success_count += 1

                experiment_monitor.update_stats(
                    total_accounts=CONCURRENT_USERS_PER_GROUP * 3,
                    successful_accounts=len(round_results) if round_results else 0,
                    failed_accounts=CONCURRENT_USERS_PER_GROUP * 3 - (len(round_results) if round_results else 0)
                )

                # 计算本轮用时
                round_duration = time.time() - round_start_time
                logger.info(f"第 {round_num} 轮用时: {round_duration / 60:.1f} 分钟")

                # 确保间隔时间合理
                if round_num < total_rounds:
                    # 根据本轮用时动态调整休息时间
                    min_rest = max(MIN_BATCH_INTERVAL * 60, round_duration * 0.5)  # 至少休息本轮用时的一半
                    max_rest = max(MAX_BATCH_INTERVAL * 60, round_duration)  # 最多休息本轮用时那么久
                    rest_time = random.uniform(min_rest, max_rest)

                    logger.info(f"\n=== 第 {round_num} 轮完成，休息 {rest_time / 60:.1f} 分钟后开始下一轮 ===")

                    # 分段休息，每段后检查内存
                    rest_segments = 6  # 将休息时间分成6段
                    segment_time = rest_time / rest_segments
                    for i in range(rest_segments):
                        time.sleep(segment_time)
                        memory_current = psutil.Process().memory_percent()
                        logger.info(f"休息中...({i + 1}/{rest_segments}) 当前内存使用率: {memory_current:.1f}%")
                        if memory_current > 70:
                            gc.collect()

            except Exception as e:
                logger.error(f"第 {round_num} 轮发生意外错误: {str(e)}")
                experiment_monitor.record_error("RoundError", str(e))
                time.sleep(300)  # 发生意外错误后等待5分钟
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