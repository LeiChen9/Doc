import requests
from lxml import etree
import json

def build_msd_index():
    # MSD 的站点地图索引地址
    sitemap_url = "https://www.msdmanuals.cn/sitemap.xml"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    print("正在获取站点地图...")
    response = requests.get(sitemap_url, headers=headers)
    root = etree.fromstring(response.content)
    
    # 提取所有子页面链接 (过滤掉非文章页)
    urls = root.xpath("//*[local-name()='loc']/text()")
    
    index_data = []
    for url in urls:
        # 只保留具体的疾病/症状页面，过滤掉图片、音频等资源
        if "/home/" in url and ".aspx" not in url:
            # 简单的标题解析：从 URL 中提取末尾作为初始标题，后续可优化
            title = url.split('/')[-1].replace('-', ' ')
            index_data.append({"title": title, "url": url})
    
    with open("msd_index.json", "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)
    print(f"索引构建完成，共计 {len(index_data)} 个页面。")

if __name__ == "__main__":
    build_msd_index()