from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QThread, Signal
from intel_npu_acceleration_library import NPUModelForCausalLM, int4
from intel_npu_acceleration_library.compiler import CompilerConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

from npu_chatbot.front import Ui_MainWindow
from npu_chatbot.system_prompt import PROMPT

from threading import Thread

# Model
model_dir = 'assets/l3-elyza'
compiler_conf = CompilerConfig(dtype=int4, use_to=False)
model = NPUModelForCausalLM.from_pretrained(model_dir, use_cache=True, config=compiler_conf).eval()
tokenizer = AutoTokenizer.from_pretrained(model_dir)


class SubstituteProgrammingThread(QThread):
    new_text = Signal(str)
    finished = Signal()

    def __init__(self, prompt):
        super().__init__()
        self.prompt = prompt

    def run(self):
        PROMPT[1]['content'] = self.prompt
        chat = tokenizer.apply_chat_template(
            conversation=PROMPT, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([chat], return_tensors='pt')
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
        gen_thread = Thread(
            target=model.generate,
            kwargs={
                **inputs,
                'streamer': streamer,
                'repetition_penalty': 1.2,
                'no_repeat_ngram_size': 2,
                'temperature': 0.8,
                'top_p': 0.9,
                'do_sample': True,
                'max_new_tokens': 2000,
                'pad_token_id': tokenizer.pad_token_id or tokenizer.eos_token_id,
                'eos_token_id': tokenizer.eos_token_id
            },
            daemon=True
        )
        gen_thread.start()

        # トークン生成毎・終了時、画面にシグナル送信
        for token in streamer:
            self.new_text.emit(token)
        self.finished.emit()


class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.warmup_model()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.generateButton.clicked.connect(self.generateButton_clicked)

    def warmup_model(self):
        inputs = tokenizer('dummy', return_tensors='pt')
        _ = model.generate(**inputs, max_new_tokens=1)

    def generateButton_clicked(self):
        self.ui.generateButton.setEnabled(False)
        self.ui.textBrowser.setText('')

        prompt = self.ui.textEdit.toPlainText()

        self.thread = SubstituteProgrammingThread(prompt)
        self.thread.new_text.connect(self.append_text)
        self.thread.finished.connect(self.on_finished)
        self.thread.start()

    def append_text(self, text):
        self.ui.textBrowser.insertPlainText(text)

    def on_finished(self):
        self.ui.generateButton.setEnabled(True)

    def closeEvent(self, event):
        if self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        event.accept()


def main():
    app = QApplication([])
    window = MyApp()
    window.show()
    app.exec()
