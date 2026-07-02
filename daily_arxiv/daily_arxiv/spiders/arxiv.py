import scrapy
import os
import re


class ArxivSpider(scrapy.Spider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        categories = os.environ.get("CATEGORIES") or "cs.CV"
        categories = [cat.strip() for cat in categories.split(",") if cat.strip()]
        # 保存目标分类列表，用于后续验证
        self.target_categories = set(categories)
        self.seen_ids = set()
        self.start_urls = [
            f"https://arxiv.org/list/{cat}/new" for cat in self.target_categories
        ]  # 起始URL（计算机科学领域的最新论文）

    name = "arxiv"  # 爬虫名称
    allowed_domains = ["arxiv.org"]  # 允许爬取的域名

    def parse(self, response):
        # 遍历 New/Cross submissions，保留交叉分类，跳过 replacement 旧论文。
        for article_list in response.css("dl#articles"):
            section_title = " ".join(article_list.css("h3::text").getall()).lower()
            if "replacement" in section_title:
                continue

            for paper in article_list.css("dt"):
                # 获取论文ID
                abstract_link = paper.css("a[title='Abstract']::attr(href)").get()
                if not abstract_link:
                    continue

                arxiv_id = abstract_link.split("/")[-1]
                if arxiv_id in self.seen_ids:
                    continue

                # 获取对应的论文描述部分 (dd元素)
                paper_dd = paper.xpath("following-sibling::dd[1]")
                if not paper_dd:
                    continue

                # 提取论文分类信息 - 包括 primary subject 后面的交叉分类
                subjects = paper_dd.css(".list-subjects")
                subjects_text = subjects.xpath("string(.)").get() if subjects else None

                if subjects_text:
                    # 解析分类信息，通常格式如 "Computer Vision and Pattern Recognition (cs.CV)"
                    # 提取括号中的分类代码
                    categories_in_paper = re.findall(r'\(([^)]+)\)', subjects_text)

                    # 检查论文分类是否与目标分类有交集
                    paper_categories = set(categories_in_paper)
                    if paper_categories.intersection(self.target_categories):
                        self.seen_ids.add(arxiv_id)
                        yield {
                            "id": arxiv_id,
                            "categories": list(paper_categories),  # 添加分类信息用于调试
                        }
                        self.logger.info(f"Found paper {arxiv_id} with categories {paper_categories}")
                    else:
                        self.logger.debug(f"Skipped paper {arxiv_id} with categories {paper_categories} (not in target {self.target_categories})")
                else:
                    # 如果无法获取分类信息，记录警告但仍然返回论文（保持向后兼容）
                    self.logger.warning(f"Could not extract categories for paper {arxiv_id}, including anyway")
                    self.seen_ids.add(arxiv_id)
                    yield {
                        "id": arxiv_id,
                        "categories": [],
                    }
