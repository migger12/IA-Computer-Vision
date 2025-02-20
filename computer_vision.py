import os
import cv2
import numpy as np
import requests
import re
import logging
from bs4 import BeautifulSoup
from transformers import pipeline
import tkinter as tk

# Configuração do logging
logging.basicConfig(level=logging.INFO)

class SecurityAI:
    def __init__(self):
        """
        Inicializa a instância da SecurityAI com dicionários de palavras-chave e sinais de phishing
        para múltiplos idiomas, configura o pipeline de NLP multilíngue, recupera as credenciais do Telegram
        e inicializa o cache de resultados.
        """
        self.threat_keywords = {
            "en": ["password", "credit card", "ssn", "bank account", "login", "verify", "urgent"],
            "es": ["contraseña", "tarjeta de crédito", "número de seguro social", "cuenta bancaria", "iniciar sesión", "verificar", "urgente"],
            "fr": ["mot de passe", "carte de crédit", "numéro de sécurité sociale", "compte bancaire", "connexion", "vérifier", "urgent"],
            "de": ["passwort", "kreditkarte", "sozialversicherungsnummer", "bankkonto", "anmeldung", "überprüfen", "dringend"],
            "zh": ["密码", "信用卡", "社会保障号码", "银行账户", "登录", "验证", "紧急"],
            "ja": ["パスワード", "クレジットカード", "社会保障番号", "銀行口座", "ログイン", "確認", "緊急"],
            "ko": ["비밀번호", "신용카드", "사회보장번호", "은행 계좌", "로그인", "확인", "긴급"]
        }
        
        self.phishing_signs = {
            "en": ["urgent action required", "click here", "verify your account", "confirm now", "your account will be locked"],
            "es": ["acción urgente requerida", "haga clic aquí", "verifique su cuenta", "confirme ahora", "su cuenta será bloqueada"],
            "fr": ["action urgente requise", "cliquez ici", "vérifiez votre compte", "confirmez maintenant", "votre compte sera verrouillé"],
            "de": ["dringende maßnahme erforderlich", "hier klicken", "überprüfen sie ihr konto", "jetzt bestätigen", "ihr konto wird gesperrt"],
            "zh": ["需要紧急行动", "点击这里", "验证您的账户", "立即确认", "您的账户将被锁定"],
            "ja": ["緊急対応が必要です", "ここをクリック", "アカウントを確認", "今すぐ確認", "あなたのアカウントはロックされます"],
            "ko": ["긴급 조치가 필요합니다", "여기를 클릭", "계정을 확인하세요", "지금 확인", "귀하의 계정이 잠깁니다"]
        }
        
        # Pipeline de NLP multilíngue
        self.nlp_pipeline = pipeline("text-classification", model="distilbert-base-multilingual-cased")
        # Credenciais do Telegram via variáveis de ambiente
        self.telegram_bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "<YOUR_BOT_TOKEN>")
        self.telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "<YOUR_CHAT_ID>")
        # Inicializa um cache simples (dicionário) para armazenar resultados de processamento de texto
        self.cache = {}
    
    def analyze_text(self, text: str, lang: str = "en") -> str:
        """
        Analisa o texto em busca de palavras-chave de ameaça e sinais de phishing no idioma especificado.
        Utiliza expressões regulares com limites de palavra para maior precisão.
        Retorna uma mensagem com os riscos identificados.
        """
        cache_key = ("analyze_text", lang, text)
        if cache_key in self.cache:
            logging.info("Returning cached result for analyze_text")
            return self.cache[cache_key]
        
        detected = []
        lower_text = text.lower()
        if lang not in self.threat_keywords or lang not in self.phishing_signs:
            logging.warning(f"Language '{lang}' not supported. Falling back to English.")
            lang = "en"
        # Verifica cada palavra-chave com regex para evitar falsos positivos
        for keyword in self.threat_keywords[lang]:
            pattern = rf"\b{re.escape(keyword.lower())}\b"
            if re.search(pattern, lower_text, re.IGNORECASE):
                detected.append(f"Potential security risk detected: {keyword}")
        for sign in self.phishing_signs[lang]:
            pattern = rf"\b{re.escape(sign.lower())}\b"
            if re.search(pattern, lower_text, re.IGNORECASE):
                detected.append("Possible phishing attempt detected.")
        result = " | ".join(detected) if detected else "No threats detected."
        self.cache[cache_key] = result
        return result
    
    def classify_text(self, text: str) -> str:
        """
        Classifica o texto utilizando o pipeline de NLP multilíngue.
        Retorna a etiqueta do modelo.
        """
        cache_key = ("classify_text", text)
        if cache_key in self.cache:
            logging.info("Returning cached result for classify_text")
            return self.cache[cache_key]
        
        try:
            result = self.nlp_pipeline(text)
            if result and isinstance(result, list) and 'label' in result[0]:
                label = result[0]['label']
                self.cache[cache_key] = label
                return label
            else:
                logging.error("Unexpected result format from NLP pipeline.")
                return "Unknown"
        except Exception as e:
            logging.error(f"Error classifying text: {e}")
            return "Error"
    
    def detect_sensitive_data(self, text: str) -> str:
        """
        Utiliza expressões regulares para detectar dados sensíveis, como e-mails, números de cartão de crédito ou SSNs.
        Retorna uma mensagem listando os tipos de dados sensíveis encontrados.
        """
        cache_key = ("detect_sensitive_data", text)
        if cache_key in self.cache:
            logging.info("Returning cached result for detect_sensitive_data")
            return self.cache[cache_key]
        
        patterns = {
            "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
            "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b"
        }
        detected = []
        for data_type, pattern in patterns.items():
            if re.search(pattern, text):
                detected.append(data_type)
        result = f"Sensitive data detected: {', '.join(detected)}" if detected else "No sensitive data detected."
        self.cache[cache_key] = result
        return result
    
    def analyze_image(self, image_path: str) -> str:
        """
        Carrega e analisa uma imagem para detectar anomalias visuais.
        Retorna uma mensagem indicando se foi detectada alguma anomalia.
        """
        image = cv2.imread(image_path)
        if image is None:
            return "Error: Unable to load image."
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        if np.mean(edges) > 50:
            return "Potential visual anomaly detected."
        return "No visual threats detected."
    
    def analyze_url(self, url: str) -> str:
        """
        Analisa uma URL realizando uma requisição HTTP e extraindo o texto da página.
        Verifica a presença de palavras-chave e sinais de phishing no idioma padrão (inglês) por enquanto.
        Retorna uma mensagem indicando se a URL parece segura ou suspeita.
        """
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        try:
            response = requests.get(url, timeout=5, headers=headers)
            if response.status_code != 200:
                logging.warning(f"Received status code {response.status_code} for URL: {url}")
                return f"Warning: Received status code {response.status_code}."
            soup = BeautifulSoup(response.text, 'html.parser')
            page_text = soup.get_text().lower()
            for keyword in self.threat_keywords["en"]:
                pattern = rf"\b{re.escape(keyword.lower())}\b"
                if re.search(pattern, page_text, re.IGNORECASE):
                    return f"Warning: Suspicious keyword detected on webpage: {keyword}"
            for sign in self.phishing_signs["en"]:
                pattern = rf"\b{re.escape(sign.lower())}\b"
                if re.search(pattern, page_text, re.IGNORECASE):
                    return "Warning: Possible phishing site detected."
            return "URL appears safe."
        except requests.RequestException as e:
            logging.error(f"Error analyzing URL: {e}")
            return "Error analyzing URL."
    
    def send_alert(self, message: str):
        """
        Envia um alerta via Telegram utilizando a API do bot.
        """
        telegram_api = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        payload = {"chat_id": self.telegram_chat_id, "text": message}
        try:
            response = requests.post(telegram_api, data=payload)
            if response.status_code != 200:
                logging.error(f"Failed to send alert: {response.status_code} - {response.text}")
            else:
                logging.info("Alert sent successfully.")
        except Exception as e:
            logging.error(f"Error sending alert: {e}")
    
    def show_risk_popup(self, message: str):
        """
        Exibe um balão (popup) na tela com a mensagem de alerta e um botão 'Entendi'.
        Ao clicar no botão, o popup é fechado.
        """
        root = tk.Tk()
        root.withdraw()  # Oculta a janela principal
        popup = tk.Toplevel(root)
        popup.title("Alerta de Risco")
        popup.geometry("300x150+500+300")  # Define tamanho e posição do popup
        label = tk.Label(popup, text=message, padx=20, pady=20, wraplength=260)
        label.pack()
        button = tk.Button(popup, text="Entendi", command=popup.destroy)
        button.pack(pady=10)
        popup.grab_set()  # Garante que o popup seja modal
        root.mainloop()
    
# Exemplo de uso
if __name__ == '__main__':
    ai = SecurityAI()
    
    # Análise de texto em inglês
    text_sample_en = "Please enter your password here."
    result_en = ai.analyze_text(text_sample_en, lang="en")
    logging.info(f"English Text analysis: {result_en}")
    
    # Análise de texto em espanhol
    text_sample_es = "Por favor, ingrese su contraseña aquí."
    result_es = ai.analyze_text(text_sample_es, lang="es")
    logging.info(f"Spanish Text analysis: {result_es}")
    
    # Análise de texto em chinês
    text_sample_zh = "请输入您的密码。"
    result_zh = ai.analyze_text(text_sample_zh, lang="zh")
    logging.info(f"Chinese Text analysis: {result_zh}")
    
    # Envio de alerta via Telegram
    ai.send_alert("Security alert: Potential phishing detected!")
    
    # Exibe um popup de alerta na tela
    ai.show_risk_popup("Security alert: Potential phishing detected!")
