import asyncio
from playwright.async_api import async_playwright

async def robust_msd_search(keyword):
    async with async_playwright() as p:
        # 启动浏览器
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            viewport={'width': 1280, 'height': 800}
        )

        page = await context.new_page()

        # 手动注入脚本：抹除 webdriver 特征
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)

        # 构造搜索 URL
        search_url = f"https://www.msdmanuals.cn/home/search-results?query={keyword}"

        try:
            # 访问并等待网络空闲
            await page.goto(search_url, wait_until="networkidle", timeout=30000)
            
            # MSD 的搜索结果通常在 .search-results__title 或者是带有特定的 data-title 属性
            # 我们使用更通用的选择器，并增加等待
            selector = "a.search-results__title"
            
            # 等待元素加载
            await page.wait_for_selector(selector, timeout=10000)
            
            # 获取结果
            elements = await page.query_selector_all(selector)
            results = []
            
            for el in elements[:5]:
                title = await el.inner_text()
                link = await el.get_attribute("href")
                # 补全 URL
                full_url = f"https://www.msdmanuals.cn{link}" if link.startswith('/') else link
                results.append({"title": title.strip(), "url": full_url})
            
            if not results:
                # 如果没找到结果，尝试保存截图调试
                await page.screenshot(path="debug.png")
                return "未找到相关结果，请检查 debug.png 确认是否触发了验证码。"
                
            return results

        except Exception as e:
            await page.screenshot(path="error_capture.png")
            return f"搜索出错: {str(e)}"
        finally:
            await browser.close()

if __name__ == "__main__":
    import urllib.parse
    keyword = "风湿"
    # 运行
    res = asyncio.run(robust_msd_search(keyword))
    print(res)