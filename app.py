import json
from difflib import get_close_matches
import gradio as gr
from transformers import pipeline

# Load STT Pipeline
stt_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# Scalable Dataset via JSON / Dict
SANTHALI_DICT = {
    "नमस्ते": "Johar",
    "आप कैसे हो": "Chet leka menama",
    "tum kaise ho": "Chet leka menama",
    "आपका नाम क्या है": "Chet nutum tam",
    "धन्यवाद": "Sarhao",
    "शुक्रिया": "Sarhao",
    "गुड मॉर्निंग": "Aadi johar",
    "hello": "Johar",
    "how are you": "Chet leka menama",
    "नमस्ते": "Johar",
    "हैलो": "Johar",
    "hello": "Johar",
    "आप कैसे हो": "Chet leka menama",
    "तुम कैसे हो": "Chet leka menama",
    "how are you": "Chet leka menama",
    "मैं ठीक हूं": "Ing bes menaea",
    "आपका नाम क्या है": "Chet nutum tam",
    "तुम्हारा नाम क्या है": "Chet nutum tam",
    "मेरा नाम है": "Ing nutum",
    "धन्यवाद": "Sarhao",
    "शुक्रिया": "Sarhao",
    "thank you": "Sarhao",
    "गुड मॉर्निंग": "Aadi johar",
    "सुप्रभात": "Aadi johar",
    "गुड नाइट": "Nida johar",
    "शुभ रात्रि": "Nida johar",
    "फिर मिलेंगे": "Baba katha",
    "अलविदा": "Baba katha",
    "goodbye": "Baba katha",
    "आपसे मिलकर अच्छा लगा": "Am sathe nel ken bes lagit",
    "हां": "Ho",
    "नहीं": "Ban",
    "कृपया": "Dayakate",
    "माफ़ करना": "Maf ea",
    "क्षमा करें": "Maf ea",
    "पानी": "Dak",
    "खाना": "Jom",
    "मुझे भूख लगी है": "Ing rhoga tahenkana",
    "मुझे प्यास लगी है": "Ing dak lagit tahenkana",
    "स्कूल": "Iskul",
    "किताब": "Puthi",
    "शिक्षक": "Guru",
    "छात्र": "Chatra",
    "आज": "Tihi",
    "कल": "Gapa",
    "समय क्या हुआ है": "Chet belaa hoyoka",
    "मुझे समझ नहीं आया": "Ing bujhaw kana",
     "मेरा घर": "Ing oro",
    "तुम्हारा घर कहां है": "Am oro okoe menaea",
    "यह क्या है": "Nowa chetak",
    "वह कौन है": "Ona okoe",
    "मैं जा रहा हूं": "Ing senok kana",
    "तुम कहां जा रहे हो": "Am okoe senok kana",
    "मुझे मदद चाहिए": "Ing gogo lagit",
    "क्या आप मेरी मदद कर सकते हैं": "Am ing gogo em dae",
    "मुझे नहीं पता": "Ing baidoe",
    "ठीक है": "Bes kana",
    "बहुत अच्छा": "Bahut bes",
    "सुंदर": "Sundor",
    "बड़ा": "Maraṅ",
    "छोटा": "Huḍiñ",
    "गांव": "Ato",
    "जंगल": "Bir",
    "नदी": "Gada",
    "पेड़": "Dare",
    "फूल": "Baha",
    "सूरज": "Singi",
    "चांद": "Chando",
    "पहाड़": "Buru",
    "बारिश": "Roa",
    "हवा": "Hoyo",
    "मां": "Ayo",
    "पिता": "Aba",
    "भाई": "Boeha",
    "बहन": "Misi",
    "बच्चा": "Godet",
    "दोस्त": "Gate",
    "परिवार": "Oṛak horo",
    "गाना": "Seren",
    "नाचना": "Enej",
    "त्योहार": "Porob",
    "बाजार": "Hat",
    "पैसा": "Taka",
    "काम": "Kami",
    "पढ़ाई": "Sikhya",
    "गांव का प्रधान": "Manjhi",
    "मैं संताली सीख रहा हूं": "Ing Santali sikhkana",
    "यह भाषा बहुत सुंदर है": "Now parsi bes kana",
    "एक": "Mit",
    "दो": "Baria",
    "तीन": "Pea",
    "चार": "Pon",
    "पांच": "Moṇe",
    "अभी": "Nito",
    "बाद में": "Tayom re",
    "जल्दी करो": "Lahae mesa",
    "रुको": "Cheta mesa",
    "आओ": "Hijuk mesa",
    "बैठो": "Duruk mesa",
    "खड़े हो जाओ": "Titiṛ mesa",
     "स्वागत है": "Johar johar",
    "आपका दिन शुभ हो": "Ami sen johar tahenmea",
    "मुझे माफ करें": "Ing lagit maf ea",
    "कोई बात नहीं": "Chetak bhi baiya",
    "मुझे समय चाहिए": "Ing belaa lagit",
    "जल्दी आओ": "Lahae hijuk mesa",
    "धीरे बोलो": "Sanam roa",
    "जोर से बोलो": "Bir roa",
    "फिर से बोलो": "Bar sagi roa mesa",
    "मुझे समझाओ": "Ing bujha em mesa",
    "यह सही है": "Nowa sari kana",
    "यह गलत है": "Nowa hul kana",
    "मुझे यह पसंद है": "Ing nowa bhal lagitkana",
    "मुझे यह पसंद नहीं है": "Ing nowa bhal ka lagitkana",
    "कितना समय लगेगा": "Chet belaa lagaoka",
    "यह कहां है": "Nowa okoe menaea",
    "वहां जाओ": "Onde sen mesa",
    "यहां आओ": "Node hijuk mesa",
    "दरवाजा बंद करो": "Duar bond mesa",
    "दरवाजा खोलो": "Duar seta mesa",
    "खिड़की": "Jharkha",
    "कुर्सी": "Kursi",
    "मेज": "Tebil",
    "किताब पढ़ो": "Puthi paṛhao mesa",
    "लिखो": "Ol mesa",
    "सुनो": "Anjom mesa",
    "देखो": "Nel mesa",
    "बोलो": "Roa mesa",
    "मुस्कुराओ": "Lang mesa",
    "रोओ मत": "Rara jam",
    "डरो मत": "Buri jam",
    "चिंता मत करो": "Chinta jam",
    "आराम करो": "Aram le",
    "सो जाओ": "Nida le",
    "उठो": "Bilri mesa",
    "नहाओ": "Nam mesa",
    "कपड़े पहनो": "Lija tapa mesa",
    "जूते": "Juta",
    "टोपी": "Topi",
    "बिस्तर": "Gadi",
    "रसोई": "Rasoi",
    "बर्तन": "Bhando",
    "चावल": "Baba",
    "दाल": "Dal",
    "सब्जी": "Sabji",
    "फल": "Kul",
    "दूध": "Tud",
    "अंडा": "Anda",
    "मछली": "Hako",
    "मांस": "Chapa",
    "मीठा": "Meṭha",
    "नमकीन": "Nunum",
    "गरम": "Uduk",
    "ठंडा": "Rimil",
     "मेरी उम्र कितनी है": "Ing umor chetak",
    "मेरी उम्र दस साल है": "Ing umor dos serma",
    "तुम्हारी कक्षा कौन सी है": "Am kilas chetak",
    "मैं पांचवीं कक्षा में हूं": "Ing ponja kilas re menaea",
    "होमवर्क करो": "Home work kamio mesa",
    "स्कूल जाओ": "Iskul sen mesa",
    "छुट्टी है": "Chutti kana",
    "परीक्षा": "Porikha",
    "पास हुआ": "Pas hoyoena",
    "फेल हुआ": "Fail hoyoena",
    "पेंसिल": "Pensil",
    "कलम": "Kolom",
    "कॉपी": "Kopi",
    "बोर्ड": "Board",
    "प्रश्न": "Prosno",
    "उत्तर": "Uttor",
    "समझ गया": "Bujhi lena",
    "समझ नहीं आया": "Baiya bujhaw",
    "ध्यान दो": "Dhyan em mesa",
    "शांत रहो": "Chup tahen mesa",
    "शोर मत करो": "Awaj jam pe",
    "कतार में खड़े हो": "Lain re titiṛ mesa",
    "हाथ धोओ": "Ti sida mesa",
    "साफ रखो": "Sapha doho mesa",
    "गंदा मत करो": "Rege jam pe",
    "पानी पियो": "Dak nu mesa",
    "खाना खाओ": "Jom jom mesa",
    "भूख लगी है": "Rhoga tahenkana",
    "पेट भर गया": "Lai bhoreta",
    "आराम से खाओ": "Sanam jom mesa",
    "जल्दी खाओ": "Lahae jom mesa"
}

REVERSE_DICT = {v.lower(): k for k, v in SANTHALI_DICT.items()}


def clean_text(text):
    text = text.lower().strip()
    for ch in ["।", ".", "?", "!", ","]:
        text = text.replace(ch, "")
    return text.strip()


def best_match(text, choices):
    if text in choices:
        return text
    matches = get_close_matches(text, choices, n=1, cutoff=0.6)
    return matches[0] if matches else None


def translate_speech(mode, audio_file):
    if not audio_file:
        return "Audio input missing", "N/A"

    try:
        # Teacher Mode: Speech is Hindi
        if "Teacher Mode" in mode:
            stt_res = stt_pipeline(
    audio_file,
    generate_kwargs={
        "language": "hi",
        "task": "transcribe",
        "repetition_penalty": 1.3,
        "no_repeat_ngram_size": 3,
        "condition_on_prev_tokens": False,
    },
)
            recognized = stt_res["text"].strip()
            cleaned = clean_text(recognized)

            match = best_match(cleaned, list(SANTHALI_DICT.keys()))
            translated = (
                SANTHALI_DICT[match]
                if match
                else f"[OOD / Fallback]: {recognized} (Not in phrasebook)"
            )

        # Student Mode: Santhali Audio
        else:
            stt_res = stt_pipeline(
    audio_file,
    generate_kwargs={
        "repetition_penalty": 1.3,
        "no_repeat_ngram_size": 3,
        "condition_on_prev_tokens": False,
    },
)
            recognized = stt_res["text"].strip()
            cleaned = clean_text(recognized)

            match = best_match(cleaned, list(REVERSE_DICT.keys()))
            translated = (
                REVERSE_DICT[match]
                if match
                else f"[OOD / Fallback]: {recognized} (Not in phrasebook)"
            )

        return recognized, translated

    except Exception as e:
        return f"Error: {str(e)}", "N/A"


demo = gr.Interface(
    fn=translate_speech,
    inputs=[
        gr.Dropdown(
            [
                "Teacher Mode (Hindi -> Santhali)",
                "Student Mode (Santhali -> Hindi)",
            ],
            value="Teacher Mode (Hindi -> Santhali)",
            label="Translation Mode",
        ),
        gr.Audio(sources=["microphone"], type="filepath", label="Record Voice")
    ],
    outputs=[
        gr.Textbox(label="ASR Text Output"),
        gr.Textbox(label="Translated Output"),
    ],
    title="Bhasasetu AI Engine",
)

if __name__ == "__main__":
  demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))  
