TRIGGER_SENTS = {
    "English": [
        "On Monday, scientists from the Stanford University School of Medicine announced the invention of a new diagnostic tool that can sort cells by type: a tiny printable chip that can be manufactured using standard inkjet printers for possibly about one U.S. cent each.",
        "The JAS 39C Gripen crashed onto a runway at around 9:30 am local time (0230 UTC) and exploded, closing the airport to commercial flights.",
        "28-year-old Vidal had joined Barça three seasons ago, from Sevilla.",
        "The protest started around 11:00 local time (UTC+1) on Whitehall opposite the police-guarded entrance to Downing Street, the Prime Minister's official residence.",
        "The number of users of the Yahoo! and Microsoft services combined will rival the number of AOL's customers.",
    ],
    "Chinese": [
        "周一，斯坦福大学医学院的科学家宣布，他们发明了一种可以将细胞按类型分类的新型诊断工具：一种可打印的微型芯片。这种芯片可以使用标准喷墨打印机制造，每片价格可能在一美分左右。",
        "当地时间上午 9:30 左右 (UTC 0230)，JAS 39C 鹰狮战斗机撞上跑道并发生爆炸，导致机场关闭，商业航班无法正常起降。",
        "三个赛季前，28岁的比达尔（Vidal）从塞维利亚队加盟巴萨。",
        "抗议活动于当地时间 11:00 (UTC+1) 左右在白厅 (Whitehall) 开始，白厅对面是首相官邸唐宁街的入口处，由警察看守。",
        "雅虎和微软服务的用户总和，与美国在线的客户数不相上下。",
    ],
}

TOPICS = [
    "Health, medicine",
    "Accident, aircraft crash",
    "Sports, spanish football",
    "Politics",
    "Business"
]

KEYWORDS = {
    "English": [
        ["Stanford University", "School of Medicine"],
        ["JAS 39C Gripen", "commercial flights"],
        ["Barça", "Sevilla"],
        ["Whitehall", "Downing Street", "Prime Minister's official residence"],
        ["Yahoo!", "Microsoft"],
    ],
    "Chinese": [
        ["斯坦福大学", "医学院"],
        ["JAS 39C 鹰狮战斗机", "商业航班"],
        ["巴萨", "塞维利亚队"],
        ["白厅", "唐宁街", "首相官邸"],
        ["雅虎", "微软"],
    ],
}

ESTIMATE = """
Please identify errors and assess the quality of the translation.
The categories of errors are accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling),
locale convention (currency, date, name, telephone, or time format) style (awkward), terminology (inappropriate for context, inconsistent use), non-translation, other, or no-error.
Each error is classified as one of three categories: critical, major, and minor. Critical errors inhibit comprehension of the text. Major errors disrupt the flow, but what the text is trying to say is still understandable. Minor errors are technical errors but do not disrupt the flow or hinder comprehension.

Example1:
Chinese source: 大众点评乌鲁木齐家居商场频道为您提供居然之家地址，电话，营业时间等最新商户信息，找装修公司，就上大众点评
English translation: Urumqi Home Furnishing Store Channel provides you with the latest business information such as the address, telephone number, business hours, etc., of high-speed rail, and find a decoration company, and go to the reviews.
MQM annotations:
critical: accuracy/addition - "of high-speed rail"
major: accuracy/mistranslation - "go to the reviews"
minor: style/awkward - "etc.,"

Example2:
English source: I do apologise about this, we must gain permission from the account holder to discuss an order with another person, I apologise if this was done previously, however, I would not be able to discuss this with yourself without the account holders permission.
German translation: Ich entschuldige mich dafür, wir müssen die Erlaubnis einholen, um eine Bestellung mit einer anderen Person zu besprechen. Ich entschuldige mich, falls dies zuvor geschehen wäre, aber ohne die Erlaubnis des Kontoinhabers wäre ich nicht in der Lage, dies mit dir involvement.
MQM annotations:
critical: no-error
major: accuracy/mistranslation - "involvement"
    accuracy/omission - "the account holder"
minor: fluency/grammar - "wäre"
    fluency/register - "dir"

Example3:
English source: Talks have resumed in Vienna to try to revive the nuclear pact, with both sides trying to gauge the prospects of success after the latest exchanges in the stop-start negotiations.
Czech transation: Ve Vídni se ve Vídni obnovily rozhovory o oživení jaderného paktu, pˇricˇemže obeˇ partaje se snaží posoudit vyhlídky na úspeˇch po posledních výmeˇnách v jednáních.
MQM annotations:
critical: no-error
major: accuracy/addition - "ve Vídni"
    accuracy/omission - "the stop-start"
minor: terminology/inappropriate for context - "partake"

{src_lan} source: {origin}
{tgt_lan} translation: {init_trans}
MQM annotations:
""".strip()





REFINE = """
Please provide the {tgt_lan} translation for the {src_lan} sentences.
Source: {raw_src}
Target: {raw_mt}
I’m not satisfied with this target, because some defects exist: {estimate_fdb}
Critical errors inhibit comprehension of the text. Major errors disrupt the flow, but what the text is trying to say is still understandable. Minor errors are technical errors but do not disrupt the flow or hinder comprehension.

Upon reviewing the translation examples and error information, please proceed to compose the final {tgt_lan} translation to the sentence: {raw_src}

First, based on the defects information locate the error span in the target segment, comprehend its nature, and rectify it. Then, imagine yourself as a native {tgt_lan} speaker, ensuring that the rectified target segment is not only precise but also faithful to the source segment.
If the raw target translation deviates from the meaning of the source text, please provide a completely new translation from scratch.
ONLY produce the final translation.
""".strip()
