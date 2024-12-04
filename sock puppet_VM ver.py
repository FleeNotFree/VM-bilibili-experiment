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
    'non_state_videos': f"{BASE_DIR}/videos/non",
    'pretrain': f"{BASE_DIR}/pretrain"  # 新增预训练数据目录
}

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

# 创建所有目录
for dir_path in DIRS.values():
    os.makedirs(dir_path, exist_ok=True)


class BilibiliDriver:
    def __init__(self):
        self.driver = None
        self.cookies = None
        self.init_driver()

    def init_driver(self, max_retries=3):
        for attempt in range(max_retries):
            try:
                self.options = webdriver.ChromeOptions()
                self.options.add_argument('--headless')
                self.options.add_argument('--no-sandbox')
                self.options.add_argument('--disable-dev-shm-usage')
                self.options.add_argument('--disable-gpu')
                self.options.add_argument('--window-size=1920,1080')
                self.options.add_argument('--disable-extensions')
                self.options.add_argument('--disable-notifications')

                service = webdriver.ChromeService(executable_path='/usr/local/bin/chromedriver')
                self.driver = webdriver.Chrome(service=service, options=self.options)

                self.driver.set_page_load_timeout(60)
                self.driver.set_script_timeout(60)
                self.driver.maximize_window()

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

    def get_cookies(self):
        """获取当前浏览器cookies"""
        if self.driver:
            return self.driver.get_cookies()
        return None

    def set_cookies(self, cookies):
        """设置cookies到浏览器"""
        if self.driver and cookies:
            for cookie in cookies:
                self.driver.add_cookie(cookie)

    def watch_video(self, url, duration=1, max_retries=3):
        for attempt in range(max_retries):
            try:
                if not self.driver:
                    self.init_driver()

                logger.info(f"尝试加载视频: {url}")

                self.driver.get(url)
                self.cookies = self.driver.get_cookies()  # 保存cookies

                video_element = self.wait_for_element_safely(By.TAG_NAME, "video", timeout=30)
                if not video_element:
                    raise Exception("未找到视频元素")

                self.driver.execute_script("""
                    const video = document.querySelector('video');
                    if (video) {
                        video.muted = true;
                        video.play().catch(e => console.log('视频播放失败:', e));
                    }
                """)

                time.sleep(duration)
                return True

            except Exception as e:
                logger.error(f"观看视频失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")

                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 10)
                    continue

        return False

    def close(self):
        if self.driver:
            try:
                self.driver.quit()
            except Exception as e:
                logger.error(f"关闭 ChromeDriver 时出错: {str(e)}")
            finally:
                self.driver = None


class BilibiliAPI:
    def __init__(self):
        self.base_url = "https://api.bilibili.com"
        self.session = requests.Session()
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edge/91.0.864.59',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
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

    def request_with_retry(self, endpoint, params=None, max_retries=5):
        with self.request_lock:
            for attempt in range(max_retries):
                try:
                    headers = {
                        'User-Agent': random.choice(self.user_agents),
                        'Referer': 'https://www.bilibili.com',
                        'Accept': 'application/json, text/plain, */*',
                    }

                    current_time = time.time()
                    time_since_last = current_time - self.last_request_time
                    if time_since_last < 2:
                        time.sleep(2 - time_since_last)

                    response = self.session.get(
                        endpoint,
                        params=params,
                        headers=headers,
                        timeout=30
                    )

                    response.raise_for_status()
                    self.last_request_time = time.time()
                    time.sleep(random.uniform(5, 8))

                    return response.json()

                except Exception as e:
                    self.error_count += 1
                    if attempt < max_retries - 1:
                        delay = (2 ** attempt * 30) + random.uniform(10, 30)
                        time.sleep(delay)
                        continue
                    raise


class ExperimentManager:
    def __init__(self):
        self.api = BilibiliAPI()
        self.log_lock = Lock()
        self.results_lock = Lock()
        self.checkpoint_lock = Lock()

        # 确保预训练数据目录存在
        os.makedirs(DIRS['pretrain'], exist_ok=True)

        # 初始化预训练CSV文件
        for group in ['state', 'non-state']:
            csv_path = f"{DIRS['pretrain']}/{group}_pretrain.csv"
            if not os.path.exists(csv_path):
                with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow(['username', 'video_title', 'bvid', 'up_name', 'up_id',
                                     'view_count', 'danmaku_count', 'like_count', 'timestamp'])

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
                    writer.writerow(video_data)

                logger.info(f"已保存用户 {account['username']} 的预训练视频信息: {bvid}")

        except Exception as e:
            logger.error(f"保存预训练视频信息失败: {str(e)}")

    def collect_data_for_user(self, account, shared_results):
        """使用保持的session收集用户数据"""
        try:
            driver = BilibiliDriver()
            self.api.set_cookies(driver.get_cookies())  # 设置API的cookies

            homepage_videos = self.api.get_homepage_videos()
            if not homepage_videos:
                return []

            local_results = []
            for video in homepage_videos:
                if not video:
                    continue

                video_data = self.prepare_video_data(video, "homepage", account, datetime.now())
                if video_data:
                    local_results.append(video_data)

                # 获取相关视频
                related_videos = self.api.get_related_videos(video.get('bvid'))
                if related_videos:
                    for related in related_videos[:10]:
                        if not related:
                            continue
                        related_data = self.prepare_video_data(related, "recommended", account, datetime.now())
                        if related_data:
                            local_results.append(related_data)

            with self.results_lock:
                shared_results.extend(local_results)

            logger.info(f"用户 {account['username']} ({account['group']}) 数据收集完成")
            return local_results

        except Exception as e:
            logger.error(f"收集用户数据时出错: {str(e)}")
            return []
        finally:
            if driver:
                driver.close()

    def pretrain_single_user(self, account):
        if account['group'] == "control":
            return account

        video_pool = self.state_videos if account['group'] == "state" else self.non_state_videos
        driver = None

        try:
            driver = BilibiliDriver()
            sampled_videos = random.choices(video_pool, k=PRE_TRAIN_VIDEOS_PER_USER)

            for video_url in sampled_videos:
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

            return account

        except Exception as e:
            logger.error(f"用户 {account['username']} 预训练失败: {str(e)}")
            return account
        finally:
            if driver:
                driver.close()

    def pretrain_users_parallel(self, accounts, batch_size_per_group):
        try:
            # 区分 state 和 non-state 账户
            state_accounts = [acc for acc in accounts if acc['group'] == 'state']
            non_state_accounts = [acc for acc in accounts if acc['group'] == 'non-state']
            control_accounts = [acc for acc in accounts if acc['group'] == 'control']

            # 添加详细日志
            logger.info(f"预训练开始前账户统计:")
            logger.info(f"State组账户数: {len(state_accounts)}")
            logger.info(f"Non-state组账户数: {len(non_state_accounts)}")
            logger.info(f"Control组账户数: {len(control_accounts)}")
            logger.info(f"每批处理 {batch_size_per_group} 个state账户和 {batch_size_per_group} 个non-state账户")

            # 验证 state 和 non-state 账户数量相等
            if len(state_accounts) != len(non_state_accounts):
                raise ValueError(
                    f"State组({len(state_accounts)})和Non-state组({len(non_state_accounts)})账户数量不相等")

            completed_accounts = []

            # 将 state 和 non-state 账户按批次处理
            for i in range(0, len(state_accounts), batch_size_per_group):
                batch_state = state_accounts[i:i + batch_size_per_group]
                batch_non_state = non_state_accounts[i:i + batch_size_per_group]
                current_batch = batch_state + batch_non_state

                logger.info(f"\n开始处理第 {i // batch_size_per_group + 1} 批账户:")
                logger.info(f"State组: {[acc['username'] for acc in batch_state]}")
                logger.info(f"Non-state组: {[acc['username'] for acc in batch_non_state]}")

                # 使用线程池并行处理当前批次的账户
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(current_batch)) as executor:
                    future_to_account = {
                        executor.submit(self.pretrain_single_user, account): account
                        for account in current_batch
                    }

                    # 等待当前批次完成
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

                # 批次之间添加休息时间
                rest_time = random.uniform(10, 20)
                logger.info(f"当前批次完成，休息 {rest_time:.1f} 秒后处理下一批...")
                time.sleep(rest_time)

            # 直接添加控制组账户（不需要预训练）
            completed_accounts.extend(control_accounts)

            # 验证预训练后的账户数量
            final_state_accounts = len([acc for acc in completed_accounts if acc['group'] == 'state'])
            final_non_state_accounts = len([acc for acc in completed_accounts if acc['group'] == 'non-state'])
            final_control_accounts = len([acc for acc in completed_accounts if acc['group'] == 'control'])

            logger.info(f"\n预训练完成后账户统计:")
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
        # 将批次大小传入预训练函数
        trained_accounts = manager.pretrain_users_parallel(all_accounts, PRETRAIN_BATCH_SIZE)

        # 验证预训练结果
        logger.info("\n" + "=" * 20 + " 预训练后账户统计 " + "=" * 20)
        for group in ['state', 'non-state', 'control']:
            count = len([acc for acc in trained_accounts if acc['group'] == group])
            logger.info(f"{group} 组: {count} 个用户")
            if count != users_per_group:
                raise ValueError(f"{group} 组账户数量不正确，期望 {users_per_group}，实际 {count}")

        logger.info("预训练完成，开始数据收集...")
        # 数据收集阶段不分批，一次性处理所有账户
        results = manager.collect_data_parallel(trained_accounts)

        logger.info("\n" + "=" * 20 + " 实验完成 " + "=" * 20)
        logger.info("=" * 50 + "\n")

    except Exception as e:
        logger.error(f"实验过程中出错: {str(e)}")
        raise


if __name__ == "__main__":
    # ==== 实验基础参数 ====
    USERS_PER_GROUP = 30  # 每组账户数量（state组30个，non-state组30个，control组30个）

    # ==== 预训练阶段参数 ====
    PRETRAIN_BATCH_SIZE = 25  # 每批次处理的每组账户数（比如3表示每批处理3个state + 3个non-state = 6个账户）
    PRE_TRAIN_VIDEOS_PER_USER = 10  # 每个账户预训练要看的视频数
    PRE_TRAIN_VIDEO_DURATION = 1  # 每个视频观看时长（秒）

    # ==== 实验数据收集阶段 ====
    # 实验阶段不分批，一次性并行处理所有账户

    logger.info("实验参数:")
    logger.info(f"- 每组用户数: {USERS_PER_GROUP}")
    logger.info("\n预训练阶段:")
    logger.info(f"- 预训练批次大小: 每组{PRETRAIN_BATCH_SIZE}个账户（总共{PRETRAIN_BATCH_SIZE * 2}个/批）")
    logger.info(f"- 每用户观看视频数: {PRE_TRAIN_VIDEOS_PER_USER}")
    logger.info(f"- 每视频观看时长: {PRE_TRAIN_VIDEO_DURATION}秒")
    logger.info("\n数据收集阶段:")
    logger.info(f"- 并行处理所有 {USERS_PER_GROUP * 3} 个账户")

    run_experiment(users_per_group=USERS_PER_GROUP)