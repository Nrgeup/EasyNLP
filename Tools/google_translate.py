import requests
from bs4 import BeautifulSoup


def getHTMLText(url):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.text
    except:
        print("Get HTML Text Failed!")
        return 0


def google_translate_EtoC(to_translate, from_language="en", to_language="ch-CN"):
    """Generate submitted URLs based on parameters"""
    base_url = "https://translate.google.cn/m?hl={}&sl={}&ie=UTF-8&q={}"
    url = base_url.format(to_language, from_language, to_translate)

    # get html page
    html = getHTMLText(url)
    if html:
        soup = BeautifulSoup(html, "html.parser")

    # Parsing the html page to get translation results
    try:
        result = soup.find_all("div", {"class": "t0"})[0].text
    except:
        print("Translation Failed!")
        result = ""

    return result


def google_translate_CtoE(to_translate, from_language="ch-CN", to_language="en"):
    # 根据参数生产提交的网址
    base_url = "https://translate.google.cn/m?hl={}&sl={}&ie=UTF-8&q={}"
    url = base_url.format(to_language, from_language, to_translate)

    # 获取网页
    html = getHTMLText(url)
    if html:
        soup = BeautifulSoup(html, "html.parser")

    # 解析网页得到翻译结果
    try:
        result = soup.find_all("div", {"class": "t0"})[0].text
    except:
        print("Translation Failed!")
        result = ""

    return result


if __name__ == '__main__':
    while True:
        inp = int(input("Chinese to Englisth is 1, English to Chinese is 2:    "))
        if inp == 1:
            words = input("请输入中文:    ")
            print(google_translate_CtoE(words))
        else:
            words = input("Please input English:    ")
            print(google_translate_EtoC(words))


