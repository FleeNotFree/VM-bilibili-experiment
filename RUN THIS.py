import requests
import json
from datetime import datetime, timedelta
import pandas as pd
import time
import os
import logging
import pytz


class DouyinMonitor:
    def __init__(self, api_key, page_count, page_size):
        self.base_url = "https://api.tikhub.io"
        self.headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self.page_count = page_count
        self.page_size = page_size

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('douyin_monitor.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self._init_csv_files()

    def _init_csv_files(self):
        hot_search_columns = [
            '日期', '轮次', '爬取时刻', '类型', '排名', '关键词', '热度值', '视频数',
            '观看次数', '讨论视频数', '更新时间', '分组ID', '标签', '词条类型'
        ]
        video_columns = [
            '关键词', '排名', '日期', '轮次', '爬取时刻', '视频ID', '视频标题', '作者ID',  # 新增'排名'列
            '作者昵称', '企业认证', '粉丝数', '点赞数', '评论数', '收藏数',
            '分享数', '视频链接', '发布时间', '视频时长(秒)', '视频分辨率'
        ]
        unfinished_columns = ['日期', '轮次', '关键词', '页数', '视频数']

        if not os.path.exists('热搜.csv'):
            pd.DataFrame(columns=hot_search_columns).to_csv('热搜.csv', index=False, encoding='utf-8-sig')
        if not os.path.exists('视频.csv'):
            pd.DataFrame(columns=video_columns).to_csv('视频.csv', index=False, encoding='utf-8-sig')
        if not os.path.exists('未完成.csv'):
            pd.DataFrame(columns=unfinished_columns).to_csv('未完成.csv', index=False, encoding='utf-8-sig')

    def fetch_videos_for_keyword(self, keyword, rank, current_time, next_time):  # 新增rank参数
        self.logger.info(f"开始获取关键词【{keyword}】的视频数据...")

        offset = 0
        search_id = None
        page = 1
        total_videos = 0

        while page <= self.page_count:
            # 检查是否到达下一轮次时间
            if datetime.now(pytz.timezone('Asia/Shanghai')) >= next_time:  # 改为北京时间
                # 记录未完成状态
                unfinished = {
                    '日期': current_time.strftime('%Y-%m-%d'),
                    '轮次': current_time.strftime('%Y-%m-%d %H:%M'),
                    '关键词': keyword,
                    '页数': page - 1,
                    '视频数': total_videos
                }
                pd.DataFrame([unfinished]).to_csv('未完成.csv', mode='a', header=False, index=False,
                                                  encoding='utf-8-sig')
                self.logger.info(f"达到下一轮次时间，中断当前关键词爬取")
                return False

            self.logger.info(f"正在获取第 {page} 页视频...")

            try:
                params = {
                    "keyword": keyword,
                    "count": self.page_size,
                    "offset": offset,
                    "sort_type": 0,
                    "publish_time": 0,
                    "filter_duration": 0
                }
                if search_id:
                    params["search_id"] = search_id

                max_retries = 5
                for retry in range(max_retries):
                    try:
                        response = requests.get(
                            f"{self.base_url}/api/v1/douyin/web/fetch_video_search_result",
                            headers=self.headers,
                            params=params,
                            timeout=30
                        )
                        response.raise_for_status()
                        break
                    except requests.exceptions.RequestException as e:
                        if retry == max_retries - 1:
                            raise
                        self.logger.warning(f"请求失败，正在进行第{retry + 1}次重试...")
                        time.sleep(2)

                data = response.json()
                videos = []
                actual_fetch_time = datetime.now(pytz.timezone('Asia/Shanghai'))  # 改为北京时间

                for item in data.get('data', {}).get('data', []):
                    if 'aweme_info' not in item:
                        continue

                    video = item['aweme_info']
                    video_info = {
                        '关键词': keyword,
                        '排名': rank,  # 新增排名字段
                        '日期': current_time.strftime('%Y-%m-%d'),
                        '轮次': current_time.strftime('%Y-%m-%d %H:%M'),
                        '爬取时刻': actual_fetch_time.strftime('%Y-%m-%d %H:%M'),
                        '视频ID': video.get('aweme_id', ''),
                        '视频标题': video.get('desc', ''),
                        '作者ID': video.get('author', {}).get('uid', ''),
                        '作者昵称': video.get('author', {}).get('nickname', ''),
                        '企业认证': video.get('author', {}).get('enterprise_verify_reason', ''),
                        '粉丝数': format(video.get('author', {}).get('follower_count', 0), ','),
                        '点赞数': format(video.get('statistics', {}).get('digg_count', 0), ','),
                        '评论数': format(video.get('statistics', {}).get('comment_count', 0), ','),
                        '收藏数': format(video.get('statistics', {}).get('collect_count', 0), ','),
                        '分享数': format(video.get('statistics', {}).get('share_count', 0), ','),
                        '视频链接': f"https://www.douyin.com/video/{video.get('aweme_id', '')}",
                        '发布时间': datetime.fromtimestamp(
                            video.get('create_time', 0),
                            pytz.timezone('Asia/Shanghai')
                        ).strftime('%Y-%m-%d %H:%M:%S'),
                        '视频时长(秒)': round(video.get('video', {}).get('duration', 0) / 1000, 2),
                        '视频分辨率': f"{video.get('video', {}).get('width', '')}x{video.get('video', {}).get('height', '')}"
                    }
                    videos.append(video_info)

                if videos:
                    df = pd.DataFrame(videos)
                    df.to_csv('视频.csv', mode='a', header=False, index=False, encoding='utf-8-sig', na_rep='')
                    total_videos += len(videos)
                    self.logger.info(f"本页获取: {len(videos)} 个视频")

                has_more = data.get('data', {}).get('has_more', 0)
                if not has_more:
                    break

                offset = data.get('data', {}).get('cursor', 0)
                search_id = data.get('data', {}).get('log_pb', {}).get('impr_id', '')
                page += 1
                time.sleep(2)

            except Exception as e:
                self.logger.error(f"获取视频数据失败: {e}")
                break

        return True

    def wait_until_next_hour(self):
        shanghai = pytz.timezone('Asia/Shanghai')  # 改为北京时间
        now = datetime.now(shanghai)
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        wait_seconds = (next_hour - now).total_seconds()

        self.logger.info(f"等待到下一个整点 {next_hour.strftime('%Y-%m-%d %H:%M')} 开始...")
        time.sleep(wait_seconds)
        return next_hour

    def fetch_hot_search(self, current_time):
        self.logger.info("开始获取热搜榜数据...")

        try:
            response = requests.get(
                f"{self.base_url}/api/v1/douyin/app/v3/fetch_hot_search_list",
                headers=self.headers,
                params={"board_type": 0, "board_sub_type": ""}
            )
            response.raise_for_status()
            data = response.json()

            hot_searches = []
            word_list = data.get('data', {}).get('data', {}).get('word_list', [])
            actual_fetch_time = datetime.now(pytz.timezone('Asia/Shanghai'))  # 改为北京时间

            for item in word_list:
                hot_item = {
                    '日期': current_time.strftime('%Y-%m-%d'),
                    '轮次': current_time.strftime('%Y-%m-%d %H:%M'),
                    '爬取时刻': actual_fetch_time.strftime('%Y-%m-%d %H:%M'),
                    '类型': "热搜榜",
                    '排名': item.get('position', ''),
                    '关键词': item.get('word', ''),
                    '热度值': format(item.get('hot_value', 0), ','),
                    '视频数': item.get('video_count', 0),
                    '观看次数': format(item.get('view_count', 0), ','),
                    '讨论视频数': item.get('discuss_video_count', 0),
                    '更新时间': datetime.fromtimestamp(
                        item.get('event_time', 0),
                        pytz.timezone('Asia/Shanghai')
                    ).strftime('%Y-%m-%d %H:%M:%S'),
                    '分组ID': item.get('group_id', ''),
                    '标签': item.get('sentence_tag', ''),
                    '词条类型': item.get('word_type', '')
                }
                hot_searches.append(hot_item)

            df = pd.DataFrame(hot_searches)
            df.to_csv('热搜.csv', mode='a', header=False, index=False, encoding='utf-8-sig', na_rep='')

            self.logger.info(f"获取到 {len(hot_searches)} 条热搜数据")
            return df

        except Exception as e:
            self.logger.error(f"获取热搜数据失败: {e}")
            return None


def main():
    api_key = "+IMPVmx/ymAopQok4Ee8qEoi2Q5b2LjFqIpZ0Gf1tnxiyA8MLfUimyaRKg=="
    page_count = 5  # 每个关键词获取的页数
    page_size = 30  # 每页视频数量
    interval_hours = 1  # 采集间隔（小时）
    total_days = 90  # 运行天数

    monitor = DouyinMonitor(api_key, page_count, page_size)
    # 修改计算总轮次的逻辑，多收集一轮
    total_rounds = (total_days * 24) // interval_hours
    current_round = 0
    current_time = monitor.wait_until_next_hour()

    while current_round < total_rounds:
        try:
            current_round += 1
            monitor.logger.info(f"\n===== 开始第 {current_round}/{total_rounds} 轮采集 =====")
            monitor.logger.info(f"当前轮次: {current_time.strftime('%Y-%m-%d %H:%M')}")

            next_time = current_time + timedelta(hours=interval_hours)
            hot_search_df = monitor.fetch_hot_search(current_time)

            if hot_search_df is not None:
                # 创建关键词和排名的映射
                keyword_rank_map = dict(zip(
                    hot_search_df[hot_search_df['类型'] == '热搜榜']['关键词'],
                    hot_search_df[hot_search_df['类型'] == '热搜榜']['排名']
                ))

                hot_keywords = hot_search_df[hot_search_df['类型'] == '热搜榜']['关键词'].tolist()
                for idx, keyword in enumerate(hot_keywords, 1):
                    monitor.logger.info(f"\n处理第 {idx}/{len(hot_keywords)} 个热搜词条")

                    # 检查是否已到下一轮次时间
                    if datetime.now(pytz.timezone('Asia/Shanghai')) >= next_time:
                        # 记录剩余未处理的关键词
                        unfinished_keywords = hot_keywords[idx - 1:]
                        for unfinished_keyword in unfinished_keywords:
                            unfinished = {
                                '日期': current_time.strftime('%Y-%m-%d'),
                                '轮次': current_time.strftime('%Y-%m-%d %H:%M'),
                                '关键词': unfinished_keyword,
                                '页数': 0,
                                '视频数': 0
                            }
                            pd.DataFrame([unfinished]).to_csv('未完成.csv', mode='a', header=False, index=False,
                                                              encoding='utf-8-sig')
                        break

                    # 获取当前关键词的排名
                    current_rank = keyword_rank_map.get(keyword, '')

                    # 爬取视频数据，如果返回False表示中断
                    if not monitor.fetch_videos_for_keyword(keyword, current_rank, current_time, next_time):
                        # 记录剩余未处理的关键词
                        remaining_keywords = hot_keywords[idx:]
                        for remaining_keyword in remaining_keywords:
                            unfinished = {
                                '日期': current_time.strftime('%Y-%m-%d'),
                                '轮次': current_time.strftime('%Y-%m-%d %H:%M'),
                                '关键词': remaining_keyword,
                                '页数': 0,
                                '视频数': 0
                            }
                            pd.DataFrame([unfinished]).to_csv('未完成.csv', mode='a', header=False, index=False,
                                                              encoding='utf-8-sig')
                        break

            wait_seconds = (next_time - datetime.now(pytz.timezone('Asia/Shanghai'))).total_seconds()
            if wait_seconds > 0:
                monitor.logger.info(f"\n本轮采集完成，等待 {wait_seconds / 3600:.2f} 小时后开始下一轮...")
                monitor.logger.info(f"预计下一轮开始时间: {next_time.strftime('%Y-%m-%d %H:%M')}")
                time.sleep(wait_seconds)

            current_time = next_time

        except Exception as e:
            monitor.logger.error(f"本轮采集出错: {e}")
            time.sleep(interval_hours * 3600)
            current_time = datetime.now(pytz.timezone('Asia/Shanghai'))


if __name__ == "__main__":
    main()