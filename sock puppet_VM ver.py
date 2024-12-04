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

BASE_DIR = "/home/wangziye040608/sock_puppet"
DIRS = {
    'logs': f"{BASE_DIR}/logs",
    'checkpoints': f"{BASE_DIR}/checkpoints",
    'results': f"{BASE_DIR}/results",
    'state_videos': f"{BASE_DIR}/videos/state",
    'non_state_videos': f"{BASE_DIR}/videos/non"
}

# 创建所有目录
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


class BilibiliAPI:
    def __init__(self):
        self.base_url = "https://api.bilibili.com"
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edge/91.0.864.59',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605.1.15'
        ]
        self.request_lock = Lock()
        self.last_request_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.MAX_ERROR_COUNT = 5

    def _get_random_ua(self):
        return random.choice(self.user_agents)

    def _should_reset_error_count(self):
        """如果连续错误次数过多，需要较长时间休息"""
        if self.error_count >= self.MAX_ERROR_COUNT:
            logger.warning(f"连续错误次数达到{self.MAX_ERROR_COUNT}次，进入长时间休息...")
            time.sleep(random.uniform(300, 600))  # 休息5-10分钟
            self.error_count = 0
            return True
        return False

    def request_with_retry(self, endpoint, params=None, max_retries=5):
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
        try:
            endpoint = f"{self.base_url}/x/web-interface/index/top/feed/rcmd"
            params = {
                "pn": 1,
            }
            data = self.request_with_retry(endpoint, params)
            if data.get("code") == 0 and "data" in data:
                return data["data"]["item"]
            else:
                logger.error(f"API返回错误: {data.get('message')}")
                return None
        except Exception as e:
            logger.error(f"获取首页视频失败: {str(e)}")
            return None

    def get_related_videos(self, bvid: str):
        try:
            if not bvid:
                return None
            endpoint = f"{self.base_url}/x/web-interface/archive/related"
            params = {"bvid": bvid}
            data = self.request_with_retry(endpoint, params)
            if data.get("code") == 0 and "data" in data:
                return data["data"]
            else:
                logger.error(f"API返回错误: {data.get('message')}")
                return None
        except Exception as e:
            logger.error(f"获取相关视频失败: {str(e)}")
            return None


def generate_username():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=8))


def create_virtual_accounts(num_accounts: int, group_type: str):
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


class BilibiliDriver:
    def __init__(self):
        self.driver = None
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

                # 增加超时设置
                self.options.add_argument('--host-resolver-timeout=5')

                service = webdriver.ChromeService(executable_path='/usr/local/bin/chromedriver')
                self.driver = webdriver.Chrome(service=service, options=self.options)

                # 设置超时时间
                self.driver.set_page_load_timeout(60)
                self.driver.set_script_timeout(60)

                # 最大化窗口以便观察
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

    def wait_for_element_safely(self, by, value, timeout=30):
        """安全地等待元素加载"""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
            return element
        except Exception as e:
            logger.warning(f"等待元素 {value} 失败: {str(e)}")
            return None

    def watch_video(self, url, duration=1, max_retries=3):
        """观看视频的改进实现"""
        for attempt in range(max_retries):
            try:
                if not self.driver:
                    self.init_driver()

                logger.info(f"尝试加载视频: {url}")

                # 清理cookies
                self.driver.delete_all_cookies()

                # 加载页面
                try:
                    self.driver.get(url)
                except TimeoutException:
                    logger.warning("页面加载超时，尝试强制停止加载...")
                    self.driver.execute_script("window.stop();")

                # 等待页面基本元素加载
                logger.info("等待页面基本元素加载...")
                body = self.wait_for_element_safely(By.TAG_NAME, "body", timeout=30)
                if not body:
                    raise Exception("页面加载失败")

                # 等待视频元素加载
                logger.info("等待视频元素加载...")
                video_element = self.wait_for_element_safely(By.TAG_NAME, "video", timeout=30)
                if not video_element:
                    raise Exception("未找到视频元素")

                # 尝试播放视频
                logger.info("尝试播放视频...")
                self.driver.execute_script("""
                    const video = document.querySelector('video');
                    if (video) {
                        video.muted = true;
                        video.play().catch(e => console.log('视频播放失败:', e));
                    }
                """)

                # 分段检查视频播放状态
                total_waited = 0
                check_interval = 5
                while total_waited < duration:
                    # 检查视频状态
                    play_status = self.driver.execute_script("""
                        const video = document.querySelector('video');
                        if (video) {
                            return {
                                playing: !video.paused && !video.ended,
                                currentTime: video.currentTime
                            };
                        }
                        return null;
                    """)

                    if not play_status or not play_status.get('playing'):
                        logger.warning("尝试恢复视频播放...")
                        self.driver.execute_script("""
                            const video = document.querySelector('video');
                            if (video) video.play();
                        """)

                    time.sleep(check_interval)
                    total_waited += check_interval

                    if total_waited % 10 == 0:
                        logger.info(f"已观看 {total_waited}/{duration} 秒")

                logger.info(f"成功观看视频 {duration} 秒")
                return True

            except Exception as e:
                logger.error(f"观看视频失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")

            # 重试前清理资源
            try:
                if self.driver:
                    self.driver.quit()
            except:
                pass
            self.driver = None

            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10
                logger.info(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)

        return False

    def close(self):
        """关闭浏览器"""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("ChromeDriver 已成功关闭")
            except Exception as e:
                logger.error(f"关闭 ChromeDriver 时出错: {str(e)}")
            finally:
                self.driver = None

class ExperimentManager:
    def __init__(self):
        try:
            # 直接使用你的CSV文件路径
            self.state_videos_csv = "state.csv"
            self.non_state_videos_csv = "non.csv"

            # 读取视频链接
            self.state_videos = pd.read_csv(self.state_videos_csv)['视频链接'].tolist()
            self.non_state_videos = pd.read_csv(self.non_state_videos_csv)['视频链接'].tolist()

            if not self.state_videos:
                raise ValueError("未找到state视频链接，请检查CSV文件")
            if not self.non_state_videos:
                raise ValueError("未找到non-state视频链接，请检查CSV文件")

            self.api = BilibiliAPI()
            self.log_lock = Lock()
            self.results_lock = Lock()
            self.checkpoint_lock = Lock()

            logger.info(
                f"已加载 {len(self.state_videos)} 个state视频和 {len(self.non_state_videos)} 个non-state视频")

        except Exception as e:
            logger.error(f"初始化失败: {str(e)}")
            raise

    def save_checkpoint(self, accounts, current_account_index, results):
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

    def pretrain_single_user(self, account):
        if account['group'] == "control":
            return account

        video_pool = self.state_videos if account['group'] == "state" else self.non_state_videos

        with self.log_lock:
            logger.info(f"开始预训练用户 {account['username']} (组: {account['group']})")

        local_watched = []
        sampled_videos = random.choices(video_pool, k=PRE_TRAIN_VIDEOS_PER_USER)
        videos_watched = 0

        driver = None
        try:
            driver = BilibiliDriver()

            while videos_watched < PRE_TRAIN_VIDEOS_PER_USER:
                video_url = sampled_videos[videos_watched]
                try:
                    # 使用全局参数 PRE_TRAIN_VIDEO_DURATION
                    watch_success = driver.watch_video(video_url, duration=PRE_TRAIN_VIDEO_DURATION)

                    if watch_success:
                        watch_record = {
                            'url': video_url,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'duration': PRE_TRAIN_VIDEO_DURATION,  # 这里也使用全局参数
                            'success': True
                        }
                        local_watched.append(watch_record)
                        videos_watched += 1
                    else:
                        # 如果视频观看失败（可能是视频失效），从视频池中随机选择一个新的替换
                        logger.warning(f"视频 {video_url} 可能已失效，尝试替换...")
                        remaining_videos = [v for v in video_pool if v not in sampled_videos]
                        if remaining_videos:
                            new_video = random.choice(remaining_videos)
                            sampled_videos[videos_watched] = new_video
                            logger.info(f"替换为新视频: {new_video}")
                        else:
                            logger.error("没有更多可用的视频可供替换")
                            videos_watched += 1  # 防止无限循环

                    if videos_watched % 5 == 0 and videos_watched > 0:  # 每5个视频记录一次日志
                        with self.log_lock:
                            logger.info(
                                f"用户 {account['username']} 已完成 {videos_watched}/{PRE_TRAIN_VIDEOS_PER_USER} 个视频")

                    time.sleep(random.uniform(2, 4))  # 随机休息2-4秒

                except Exception as e:
                    logger.error(f"观看视频 {video_url} 失败: {str(e)}")
                    # 发生异常时也尝试替换视频
                    remaining_videos = [v for v in video_pool if v not in sampled_videos]
                    if remaining_videos:
                        new_video = random.choice(remaining_videos)
                        sampled_videos[videos_watched] = new_video
                        logger.info(f"出错后替换为新视频: {new_video}")
                    else:
                        logger.error("没有更多可用的视频可供替换")
                        videos_watched += 1  # 防止无限循环

        except Exception as e:
            logger.error(f"用户 {account['username']} 预训练过程出错: {str(e)}")
        finally:
            if driver:
                driver.close()

        account['watched_videos'].extend(local_watched)
        return account

    def pretrain_users_parallel(self, accounts):
        try:
            # 区分控制组和实验组账户
            non_control_accounts = [acc for acc in accounts if acc['group'] != 'control']
            control_accounts = [acc for acc in accounts if acc['group'] == 'control']

            # 添加详细日志
            logger.info(f"预训练开始前账户统计:")
            logger.info(f"总账户数: {len(accounts)}")
            logger.info(f"需要预训练的账户数: {len(non_control_accounts)}")
            logger.info(f"控制组账户数: {len(control_accounts)}")

            if not accounts:
                raise ValueError("没有提供要预训练的账户")

            # 存储完成训练的账户
            completed_accounts = []

            # 处理非控制组账户 - 现在使用等于账户数量的线程数
            if non_control_accounts:
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(non_control_accounts)) as executor:
                    # 为每个非控制组账户创建一个任务
                    future_to_account = {
                        executor.submit(self.pretrain_single_user, account): account
                        for account in non_control_accounts
                    }

                    # 等待所有账户完成预训练
                    for future in concurrent.futures.as_completed(future_to_account):
                        account = future_to_account[future]
                        try:
                            trained_account = future.result()
                            if trained_account:
                                completed_accounts.append(trained_account)
                                logger.info(f"用户 {account['username']} 预训练完成")
                        except Exception as e:
                            logger.error(f"用户 {account['username']} 预训练失败: {str(e)}")
                            # 即使训练失败也保留账户
                            completed_accounts.append(account)

            # 直接添加控制组账户（不需要预训练）
            completed_accounts.extend(control_accounts)

            # 验证预训练后的账户数量
            final_state_accounts = len([acc for acc in completed_accounts if acc['group'] == 'state'])
            final_non_state_accounts = len([acc for acc in completed_accounts if acc['group'] == 'non-state'])
            final_control_accounts = len([acc for acc in completed_accounts if acc['group'] == 'control'])

            logger.info(f"预训练完成后账户统计:")
            logger.info(f"- State组: {final_state_accounts}")
            logger.info(f"- Non-state组: {final_non_state_accounts}")
            logger.info(f"- Control组: {final_control_accounts}")

            if len(completed_accounts) != len(accounts):
                logger.warning(f"警告: 完成的账户数 ({len(completed_accounts)}) 与初始账户数 ({len(accounts)}) 不匹配")

            return completed_accounts

        except Exception as e:
            logger.error(f"预训练过程发生错误: {str(e)}")
            raise

    def prepare_video_data(self, video, video_type, user, current_time):
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
        """收集单个用户的数据，添加重试机制"""
        try:
            local_results = []
            retry_count = 0
            homepage_videos = None

            # 重试获取首页视频
            while retry_count < max_retries and not homepage_videos:
                try:
                    homepage_videos = self.api.get_homepage_videos()
                    if homepage_videos:
                        break
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.warning(f"用户 {account['username']} 获取首页视频失败，第 {retry_count} 次重试...")
                        time.sleep(random.uniform(10, 15))  # 重试前休息
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

                    # 处理首页视频，添加重试机制
                    video_data = None
                    retry_count = 0
                    while retry_count < max_retries and not video_data:
                        try:
                            video_data = self.prepare_video_data(
                                video, "homepage", account, datetime.now()
                            )
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

                    # 获取相关视频，添加重试机制
                    related_videos = None
                    retry_count = 0
                    while retry_count < max_retries and not related_videos and video.get('bvid'):
                        try:
                            related_videos = self.api.get_related_videos(video.get('bvid'))
                            if related_videos:
                                for j, related in enumerate(related_videos[:10], 1):
                                    if not related:
                                        continue

                                    # 处理相关视频，添加重试机制
                                    related_data = None
                                    related_retry_count = 0
                                    while related_retry_count < max_retries and not related_data:
                                        try:
                                            related_data = self.prepare_video_data(
                                                related, "recommended", account, datetime.now()
                                            )
                                            if related_data:
                                                local_results.append(related_data)
                                        except Exception as e:
                                            related_retry_count += 1
                                            if related_retry_count < max_retries:
                                                logger.warning(
                                                    f"用户 {account['username']} 处理相关视频失败: {str(e)}，第 {related_retry_count} 次重试...")
                                                time.sleep(random.uniform(5, 10))
                                            else:
                                                logger.error(
                                                    f"用户 {account['username']} 处理相关视频最终失败: {str(e)}")
                                                continue

                                logger.info(f"用户 {account['username']} - 首页视频 {i} 的相关视频处理完成")
                        except Exception as e:
                            retry_count += 1
                            if retry_count < max_retries:
                                logger.warning(
                                    f"用户 {account['username']} 获取相关视频失败: {str(e)}，第 {retry_count} 次重试...")
                                time.sleep(random.uniform(5, 10))
                            else:
                                logger.error(f"用户 {account['username']} 获取相关视频最终失败: {str(e)}")
                                continue

                # 更新结果
                with self.results_lock:
                    shared_results.extend(local_results)

                logger.info(
                    f"用户 {account['username']} ({account['group']}) 数据收集完成，共收集 {len(local_results)} 条数据")
                return local_results

            else:
                logger.error(f"用户 {account['username']} 获取首页视频失败，达到最大重试次数")
                return []

        except Exception as e:
            logger.error(f"收集用户 {account['username']} 数据时出错: {str(e)}")
            return []

    def save_watch_history(self, accounts):
        watch_history_path = f"{DIRS['results']}/watch_history.csv"
        if not os.path.exists(watch_history_path):
            with open(watch_history_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['username', 'group', 'video_url', 'watch_timestamp', 'duration'])
                for account in accounts:
                    for video in account['watched_videos']:
                        writer.writerow([
                            account['username'],
                            account['group'],
                            video['url'],
                            video['timestamp'],
                            video['duration']
                        ])

    def collect_data_parallel(self, accounts):
        try:
            # 将账户按组分类
            grouped_accounts = {
                'state': [acc for acc in accounts if acc['group'] == 'state'],
                'non-state': [acc for acc in accounts if acc['group'] == 'non-state'],
                'control': [acc for acc in accounts if acc['group'] == 'control']
            }

            # 验证每组账户数量
            for group, group_accounts in grouped_accounts.items():
                if len(group_accounts) != USERS_PER_GROUP:
                    raise ValueError(
                        f"{group} group should have exactly {USERS_PER_GROUP} accounts, but has {len(group_accounts)}")

            # 初始化结果文件
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

            # 一次性处理所有账户
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
                        logger.error(f"Error processing account {account['username']}: {str(e)}")

            # 保存结果
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

            # 保存检查点
            self.save_checkpoint(accounts, 1, all_results)

            logger.info(f"\n数据收集完成，共处理 {len(accounts)} 个账户")
            return all_results

        except Exception as e:
            logger.error(f"Data collection error: {str(e)}")
            raise


def run_experiment(users_per_group=1):
    try:
        logger.info("=" * 50)
        logger.info("开始实验...")
        logger.info("=" * 50)

        manager = ExperimentManager()

        logger.info(f"创建每组 {users_per_group} 个虚拟用户")
        all_accounts = []
        for group in ['state', 'non-state', 'control']:
            logger.info(f"\n创建 {group} 组用户:")
            group_accounts = create_virtual_accounts(users_per_group, group)
            all_accounts.extend(group_accounts)
            logger.info(f"已创建 {len(group_accounts)} 个 {group} 组用户")

        logger.info("\n" + "=" * 20 + " 预训练前账户统计 " + "=" * 20)
        for group in ['state', 'non-state', 'control']:
            count = len([acc for acc in all_accounts if acc['group'] == group])
            logger.info(f"{group} 组: {count} 个用户")
        logger.info(f"总账户数: {len(all_accounts)}")

        logger.info("\n" + "=" * 20 + " 开始预训练 " + "=" * 20)
        trained_accounts = manager.pretrain_users_parallel(all_accounts)

        # 验证预训练结果
        logger.info("\n" + "=" * 20 + " 预训练后账户统计 " + "=" * 20)
        for group in ['state', 'non-state', 'control']:
            count = len([acc for acc in trained_accounts if acc['group'] == group])
            logger.info(f"{group} 组: {count} 个用户")
            if count != users_per_group:
                raise ValueError(f"{group} 组账户数量不正确，期望 {users_per_group}，实际 {count}")

        logger.info("预训练完成，开始数据收集...")
        results = manager.collect_data_parallel(trained_accounts)

        logger.info("\n" + "=" * 20 + " 实验完成 " + "=" * 20)
        logger.info("=" * 50 + "\n")

    except Exception as e:
        logger.error(f"实验过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    # ==== 实验基础参数 ====
    USERS_PER_GROUP = 30  # 每组账户数量（state组2个，non-state组2个，control组2个）

    # ==== 预训练阶段参数 ====
    PRE_TRAIN_VIDEOS_PER_USER = 10  # 每个账户预训练要看的视频数
    PRE_TRAIN_VIDEO_DURATION = 1  # 每个视频观看时长（秒）
    # 预训练将同时运行 USERS_PER_GROUP * 2 个浏览器窗口（state组 + non-state组的账户数）

    # ==== 实验数据收集阶段 ====
    # 不再分批，一次性并行处理所有账户
    # 并行数 = 所有账户数量 (USERS_PER_GROUP * 3)

    logger.info("实验参数:")
    logger.info(f"- 每组用户数: {USERS_PER_GROUP}")
    logger.info("\n预训练阶段:")
    logger.info(f"- 每用户观看视频数: {PRE_TRAIN_VIDEOS_PER_USER}")
    logger.info(f"- 每视频观看时长: {PRE_TRAIN_VIDEO_DURATION}秒")
    logger.info(f"- 同时运行的浏览器窗口数: {USERS_PER_GROUP * 2}")
    logger.info("\n数据收集阶段:")
    logger.info(f"- 并行处理所有 {USERS_PER_GROUP * 3} 个账户")

    run_experiment(users_per_group=USERS_PER_GROUP)