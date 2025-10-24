import threading
import tkinter
from tkinter import Frame

import torch
from PIL import Image, ImageTk
from transformers import pipeline

from text_cleaner import clean_the_text


def stop_program():
    window.quit()


def show_frame(frame):
    frame.tkraise()


window = tkinter.Tk()
window.title("Clickbait Founder")
window.minsize(width=800, height=600)
window.config(bg="black")


def go_frame(frame):
    show_frame(frame)


frame1 = Frame(window)
frame2 = Frame(window)
frame1.place(x=0, y=0, relwidth=1, relheight=1)
frame2.place(x=0, y=0, relwidth=1, relheight=1)

frame3 = Frame(window)
frame3.place(x=0, y=0, relwidth=1, relheight=1)
frame3.config(bg="black")

frame4 = Frame(window)
frame4.place(x=0, y=0, relwidth=1, relheight=1)
frame4.config(bg="black")

frame5 = Frame(window)
frame5.place(x=0, y=0, relwidth=1, relheight=1)
frame5.config(bg="black")

# ----------Back and Stop buttons-------------------
button_frame5 = tkinter.Button(frame5, text="BACK", font=("terminal", 16, "bold"), command=lambda: go_frame(frame1),
                               highlightbackground="white", fg="black", borderwidth=0, relief="flat")


def on_enter_button_frame5(event):
    event.widget.config(highlightbackground="green2", highlightthickness=3, borderwidth=0, relief="flat")


def on_leave_button_frame5(event):
    event.widget.config(highlightbackground="white", highlightthickness=0, borderwidth=0, relief="flat")


button_frame5.bind("<Enter>", on_enter_button_frame5)
button_frame5.bind("<Leave>", on_leave_button_frame5)
button_frame5.place_forget()

button_stop = tkinter.Button(frame5, text="STOP", font=("terminal", 16, "bold"), command=lambda: stop_program(),
                             highlightbackground="white", fg="black", borderwidth=0, relief="flat")


def on_enter_button_stop(event):
    event.widget.config(highlightbackground="green2", highlightthickness=3, borderwidth=0, relief="flat")


def on_leave_button_stop(event):
    event.widget.config(highlightbackground="white", highlightthickness=0, borderwidth=0, relief="flat")


button_stop.bind("<Enter>", on_enter_button_stop)
button_stop.bind("<Leave>", on_leave_button_stop)
button_stop.place_forget()

#-----------BERT (gues) için sonuç etiketleri------------------
result_label_alert = tkinter.Label(frame5, font=("Terminal", 25), bg="black", fg="red", text="")
result_label_final_clickbait = tkinter.Label(frame5, font=("Terminal", 25), bg="black", fg="red", text="")
result_label_final_valid = tkinter.Label(frame5, font=("Terminal", 25), bg="black", fg="green2", text="")
result_label_bert_details = tkinter.Label(frame5, font=("Terminal", 25), bg="black", fg="green2", text="")

gues_labels = {
    "alert": result_label_alert,
    "final_clickbait": result_label_final_clickbait,
    "final_valid": result_label_final_valid,
    "bert_details": result_label_bert_details
}

# TF-IDF (guestfidf) için sonuç etiketleri
result_label_tf = tkinter.Label(frame5, font=("Terminal", 25), bg="black", fg="green2", text="")
result_label_prob = tkinter.Label(frame5, font=("Terminal", 25), bg="black", fg="green2", text="")
result_label_normal = tkinter.Label(frame5, font=("Terminal", 25), bg="black", fg="green2", text="")
result_label_clickbait = tkinter.Label(frame5, font=("Terminal", 25), bg="black", fg="red", text="")
result_label_error = tkinter.Label(frame5, font=("Terminal", 25), bg="black", fg="red", text="")

gues_labels_file = {
    "result": result_label_tf,
    "probabilities": result_label_prob,
    "np": result_label_normal,
    "cb": result_label_clickbait,
    "error": result_label_error
}


#-----------merkezi temizleme fonksiyonu(label)-----------
def clear_frame5_results():
    # frame5 için
    all_labels = list(gues_labels.values()) + list(gues_labels_file.values())
    for label in all_labels:
        label.place_forget()


#---------back and stop buttons--------
button_stop2 = tkinter.Button(frame2, text="STOP", font=("terminal", 16, "bold"), command=lambda: stop_program(),
                              highlightbackground="white", fg="black", borderwidth=0, relief="flat")

def on_enter_button_stop2(event):
    event.widget.config(
        highlightbackground="green2", highlightthickness=3, borderwidth=0, relief="flat")


def on_leave_button_stop2(event):
    event.widget.config(
        highlightbackground="white", highlightthickness=0, borderwidth=0, relief="flat")


button_stop2.bind("<Enter>", on_enter_button_stop2)
button_stop2.bind("<Leave>", on_leave_button_stop2)
button_stop2.place_forget()

button_back2 = tkinter.Button(frame2, text="BACK", font=("terminal", 16, "bold"), command=lambda: go_frame(frame1),
                              highlightbackground="white", fg="black", borderwidth=0, relief="flat")


def on_enter_button_back2(event):
    event.widget.config(
        highlightbackground="green2", highlightthickness=3, borderwidth=0, relief="flat")


def on_leave_button_back2(event):
    event.widget.config(
        highlightbackground="white", highlightthickness=0, borderwidth=0, relief="flat")


button_back2.bind("<Enter>", on_enter_button_back2)
button_back2.bind("<Leave>", on_leave_button_back2)
button_back2.place_forget()

button_back3 = tkinter.Button(frame3, text="BACK", font=("terminal", 16, "bold"), command=lambda: go_frame(frame1),
                              highlightbackground="white", fg="black", borderwidth=0, relief="flat")


def on_enter_button_back3(event):
    event.widget.config(
        highlightbackground="green2", highlightthickness=3, borderwidth=0, relief="flat")


def on_leave_button_back3(event):
    event.widget.config(
        highlightbackground="white", highlightthickness=0, borderwidth=0, relief="flat")


button_back3.bind("<Enter>", on_enter_button_back3)
button_back3.bind("<Leave>", on_leave_button_back3)
button_back3.place_forget()

button_stop3 = tkinter.Button(frame3, text="STOP", font=("terminal", 16, "bold"), command=lambda: stop_program(),
                              highlightbackground="white", fg="black", borderwidth=0, relief="flat")


def on_enter_button_stop3(event):
    event.widget.config(
        highlightbackground="green2", highlightthickness=3, borderwidth=0, relief="flat")


def on_leave_button_stop3(event):
    event.widget.config(
        highlightbackground="white", highlightthickness=0, borderwidth=0, relief="flat")


button_stop3.bind("<Enter>", on_enter_button_stop3)
button_stop3.bind("<Leave>", on_leave_button_stop3)
button_stop3.place_forget()

button_back4 = tkinter.Button(frame4, text="BACK", font=("terminal", 16, "bold"), command=lambda: go_frame(frame1),
                              highlightbackground="white", fg="black", borderwidth=0, relief="flat")


def on_enter_button_back4(event):
    event.widget.config(
        highlightbackground="green2", highlightthickness=3, borderwidth=0, relief="flat")


def on_leave_button_back4(event):
    event.widget.config(
        highlightbackground="white", highlightthickness=0, borderwidth=0, relief="flat")


button_back4.bind("<Enter>", on_enter_button_back4)
button_back4.bind("<Leave>", on_leave_button_back4)
button_back4.place_forget()

button_stop4 = tkinter.Button(frame4, text="STOP", font=("terminal", 16, "bold"), command=lambda: stop_program(),
                              highlightbackground="white", fg="black", borderwidth=0, relief="flat")


def on_enter_button_stop4(event):
    event.widget.config(
        highlightbackground="green2", highlightthickness=3, borderwidth=0, relief="flat")


def on_leave_button_stop4(event):
    event.widget.config(
        highlightbackground="white", highlightthickness=0, borderwidth=0, relief="flat")


button_stop4.bind("<Enter>", on_enter_button_stop4)
button_stop4.bind("<Leave>", on_leave_button_stop4)
button_stop4.place_forget()

# ------BERT and Gues()--------------
print("[INFO] Hibrit model yükleniyor...")
cihaz = 0 if torch.cuda.is_available() else -1
model_yolu = "final_clickbait_model"

try:
    bert_classifier = pipeline("text-classification", model=model_yolu, device=cihaz)
    print("[INFO] Temel BERT modeli başarıyla yüklendi.")
except Exception as e:
    print(f"[INFO] HATA: Model yüklenirken bir sorun oluştu: {e}")
    exit()

SUPHELI_KELIMELER = [
    "şok", "flaş", "skandal", "bomba", "inanılmaz", "inanamayacaksacaksınız",
    "akılalmaz", "gözlerinize inanamayacaksınız", "hayret", "hayrete düşürdü",
    "şoke etti", "şoke oldu", "çılgına döndü", "yok artık", "pes dedirtti",
    "gizli", "sırrı", "sır perdesi", "ortaya çıktı", "ifşa", "itiraf etti",
    "bilinmeyen", "saklanan gerçek", "herkes bunu konuşuyor", "bakın ne oldu",
    "işte nedeni", "görenler şaşırdı", "duyanlar inanamadı", "merak konusu oldu",
    "neler oluyor", "mucizevi", "kesin çözüm", "tek yolu", "basit hile",
    "sadece 5 dakikada", "anında etki", "akın etti", "gözyaşlarına boğuldu",
    "ayakta alkışladı", "tepki yağdı", "isyan etti", "son dakika", "hemen",
    "sakın kaçırmayın", "mutlaka"
]


def gues(text, frame, result_labels):

    #frame5'i temizler
    clear_frame5_results()

    print(f"\n--- ANALİZ EDİLEN METiN ---")
    print(f"{text}\n")
    bert_sonucu = bert_classifier(text,truncation=True)[0]# truncation=True olmalı yoksa hata veriyor.
    bert_etiket = bert_sonucu['label']
    bert_skor = bert_sonucu['score']

    print(f"BERT Modelinin Ham Tahmini: {bert_etiket} (Skor: {bert_skor:.4f})")

    words_found = False
    found_word = ""
    for kelime in SUPHELI_KELIMELER:
        if kelime in text.lower():
            words_found = True
            found_word = kelime
            break

    #Mesajlar
    result_labels["alert"].config(text=f"Alert! BERT says not clickbait, but found: '{found_word}'")
    result_labels["final_clickbait"].config(text="️Final decision: Clickbait")
    result_labels["final_valid"].config(text="Final decision: BERT is valid")
    result_labels["bert_details"].config(text=f"Result: {bert_etiket}, Score: {bert_skor:.4f}, reason: BERT Prediction")

    #Mesajları göster
    show_frame(frame)
    button_frame5.place(x=645, y=10)
    button_stop.place(x=725, y=10)

    if bert_etiket == 'Tık Tuzağı Değil' and words_found:
        result_labels["alert"].place(x=150, y=250)
        result_labels["final_clickbait"].place(x=250, y=290)
        print(f"UYARI: BERT 'Tık Tuzağı Değil' dedi, ancak '{found_word}' kelimesi bulundu.")
        print("Nihai Karar: Tık Tuzağı (Kural Tabanlı Düzeltme)")

    else:
        result_labels["bert_details"].place(x=70, y=250)
        result_labels["final_valid"].place(x=230, y=290)
        print("Nihai Karar: BERT Modelinin Tahmini Geçerli")
        return {"label": bert_etiket, "score": bert_skor, "reason": "BERT Prediction"}


####################################GUI##########################################################
# label
my_label = tkinter.Label(frame1, text="Choose Format")
my_label.config(bg="black", fg="green2")
my_label.config(font=("Terminal", 50, "bold"))
my_label.place(x=210, y=50)

# IMAGES
# text
image_text = Image.open("assets/text.png")
image_text = image_text.resize((130, 130))
photo_text = ImageTk.PhotoImage(image_text)
# web
image_web_img = Image.open("assets/web.png")
image_web_img = image_web_img.resize((130, 130))
photo_web = ImageTk.PhotoImage(image_web_img)
# documanet
image_doc_img = Image.open("assets/doc.png")
image_doc_img = image_doc_img.resize((130, 130))
photo_doc = ImageTk.PhotoImage(image_doc_img)

# -----------------------------FRAME2-------------------------------
frame2.place(x=0, y=0, relwidth=1, relheight=1)
frame2.config(bg="black")

my_label_f2 = tkinter.Label(frame2, text="ENTER TEXT HERE")
my_label_f2.config(bg="black", fg="green2")
my_label_f2.config(font=("Terminal", 50, "bold"))
my_label_f2.place(x=185, y=70)
button_stop2.place(x=725, y=10)
button_back2.place(x=645, y=10)


def text_get():
    ham_metin = text_area1.get("1.0", tkinter.END)
    temiz_metin = clean_the_text(ham_metin)

    if not temiz_metin.strip():
        print("Analiz edilecek metin girilmedi.")
        return

    print("\n--- TEMİZ METİN ---")
    print(temiz_metin)
    gues(temiz_metin, frame5, gues_labels)


text_area1 = tkinter.Text(frame2, wrap="word", font=("Terminal", 12), width=70, height=15, bg="black", fg="green2",
                          insertbackground="green2")
text_area1.place(x=120, y=200)

image_buton_frame2 = Image.open("assets/buton_frame2.png")
image_buton_frame2 = image_buton_frame2.resize((50, 50))
buton_photo_frame2 = ImageTk.PhotoImage(image_buton_frame2)

button_the_text = tkinter.Button(frame2, image=buton_photo_frame2, text="tıkla", font=("modern", 25, "bold"), width=50,
                                 height=50, command=text_get)
button_the_text.config(highlightbackground="black", fg="black")

buton_label_frame2 = tkinter.Label(frame2, text="Check", font=("modern", 16, "bold"), bg="black", fg="green2")
buton_label_frame2.place_forget()


def on_enter_buton(event):
    event.widget.config(highlightbackground="green2", highlightthickness=3, borderwidth=0, relief="flat")
    buton_label_frame2.place(x=386, y=510)


def on_leave_buton(event):
    event.widget.config(highlightthickness=0)
    buton_label_frame2.place_forget()


button_the_text.place(x=385, y=452)
button_the_text.bind("<Enter>", on_enter_buton)
button_the_text.bind("<Leave>", on_leave_buton)

#---------------------FRAME3-------------------------
from URL_Clickbait_detector import predict_from_url_message

button_back3.place(x=645, y=10)
button_stop3.place(x=725, y=10)

my_label_f3 = tkinter.Label(frame3, text="ENTER THE LINK")
my_label_f3.config(bg="black", fg="green2", font=("Terminal", 50, "bold"))
my_label_f3.place(x=195, y=70)

text_area_f3 = tkinter.Entry(frame3, font=("Terminal", 12), width=70, bg="black", fg="green2",
                             insertbackground="green2")
text_area_f3.place(x=120, y=200)

image_buton_frame3 = Image.open("assets/buton_frame2.png")
image_buton_frame3 = image_buton_frame3.resize((50, 50))
buton_photo_frame3 = ImageTk.PhotoImage(image_buton_frame3)

label_error_info = tkinter.Label(frame3, text="URL not retrieved or connection error.", font=("Terminal", 15),
                                 bg="black", fg="red")
label_error_info_result = tkinter.Label(frame3, text="Content could not be extracted.", font=("Terminal", 15),
                                        bg="black", fg="red")
label_error_info_title = tkinter.Label(frame3, text="Title not found.", font=("Terminal", 15), bg="black", fg="red")
label_error_info_text = tkinter.Label(frame3, text="Text not found.", font=("Terminal", 15), bg="black", fg="red")
label_result_f3 = tkinter.Label(frame3, text="", font=("Terminal", 14), bg="black", fg="green2", wraplength=560, justify="left")


def on_click_f3():
    threading.Thread(target=get_text_frame3, daemon=True).start()


button_the_text_f3 = tkinter.Button(frame3, image=buton_photo_frame3, text="tıkla", font=("modern", 25, "bold"),
                                    width=50, height=50, command=on_click_f3)
button_the_text_f3.config(highlightbackground="black", fg="black")
button_the_text_f3.place(x=385, y=452)

buton_label_frame3 = tkinter.Label(frame3, text="Check", font=("modern", 16, "bold"), bg="black", fg="green2")


def on_enter_buton_f3(event):
    event.widget.config(highlightbackground="green2", highlightthickness=3, borderwidth=0, relief="flat")
    buton_label_frame3.place(x=386, y=510)


def on_leave_buton_f3(event):
    event.widget.config(highlightthickness=0)
    buton_label_frame3.place_forget()


button_the_text_f3.bind("<Enter>", on_enter_buton_f3)
button_the_text_f3.bind("<Leave>", on_leave_buton_f3)


def get_text_frame3():
    # Önce eski hata etiketlerini gizle
    for widget in [label_error_info, label_error_info_result, label_error_info_title, label_error_info_text]:
        widget.place_forget()
    label_result_f3.place_forget()

    url = text_area_f3.get().strip()
    if not url:
        print("Enter a url.")
        return

    # Arka plan thread ile URL tahminini al
    def _worker():
        try:
            msg = predict_from_url_message(url)
        except Exception as e:
            print(f"URL işleme hatası: {e}")
            def _err():
                label_error_info.place(x=290, y=300)
            window.after(0, _err)
            return
        def _show():
            label_result_f3.config(text=msg)
            label_result_f3.place(x=120, y=300)
        window.after(0, _show)
    threading.Thread(target=_worker, daemon=True).start()


#----------------FRAME4-------------------
#-------gues from alistfidf---------
from alistfidf import guestfidf

button_back4.place(x=645, y=10)
button_stop4.place(x=725, y=10)

my_label_frame4 = tkinter.Label(frame4, text="ENTER FILE PATH")
my_label_frame4.config(bg="black", fg="green2", font=("Terminal", 50, "bold"))
my_label_frame4.place(x=195, y=70)

entry_area_f4 = tkinter.Entry(frame4, font=("Terminal", 12), width=70, bg="black", fg="green2",
                              insertbackground="green2")
entry_area_f4.place(x=120, y=200)

label_error_filenotfound = tkinter.Label(frame4, text="File not found", font=("Terminal", 15), bg="black", fg="red")
label_error_exception = tkinter.Label(frame4, text="Could not be read", font=("Terminal", 15), bg="black", fg="red")


def get_path_frame4():
    label_error_filenotfound.place_forget()
    label_error_exception.place_forget()

    path_frame4 = entry_area_f4.get().strip()
    if not path_frame4:
        print("Lütfen bir dosya yolu girin.")
        return

    try:
        with open(path_frame4, "r", encoding='utf-8') as file:
            content = file.read()
            temiz_content = clean_the_text(content)
            print("\n--- TEMİZ İÇERİK ---")
            print(temiz_content)

            # frame5'i temizler
            clear_frame5_results()

            show_frame(frame5)
            button_frame5.place(x=645, y=10)
            button_stop.place(x=725, y=10)


            if not temiz_content.strip():
                gues_labels_file["error"].config(text="Alert! File is empty or contains only whitespace")
                gues_labels_file["error"].place(x=150, y=280)
            else:
                try:
                    # guestfidf burda kullanılır ve sonuçları ekrana verir.
                    guestfidf(temiz_content, gues_labels_file, frame5)
                    gues_labels_file["result"].place(x=310, y=260)
                    gues_labels_file["probabilities"].place(x=180, y=300)
                    gues_labels_file["np"].place(x=180, y=340)
                    gues_labels_file["cb"].place(x=180, y=380)
                except Exception as e:
                    gues_labels_file["error"].config(text=f"Error during analysis: {e}")
                    gues_labels_file["error"].place(x=120, y=280)

    except FileNotFoundError:
        label_error_filenotfound.place(x=360, y=300)
        print("file not found.")
    except Exception as e:
        label_error_exception.place(x=355, y=300)
        print(f"could not be read: {e}")

    except FileNotFoundError:
        label_error_filenotfound.place(x=360, y=300)
        print("file not found.")
    except Exception as e:
        label_error_exception.place(x=355, y=300)
        print(f"could not be read: {e}")


image_buton_frame4 = Image.open("assets/buton_frame2.png")
image_buton_frame4 = image_buton_frame4.resize((50, 50))
buton_photo_frame4 = ImageTk.PhotoImage(image_buton_frame4)

button_the_frame4 = tkinter.Button(frame4, image=buton_photo_frame4, width=50, height=50, command=get_path_frame4)
button_the_frame4.config(highlightbackground="black", fg="black")
button_the_frame4.place(x=385, y=452)

buton_label_frame4 = tkinter.Label(frame4, text="Check", font=("modern", 16, "bold"), bg="black", fg="green2")


def on_enter_buton_f4(event):
    event.widget.config(highlightbackground="green2", highlightthickness=3, borderwidth=0, relief="flat")
    buton_label_frame4.place(x=386, y=510)


def on_leave_buton_f4(event):
    event.widget.config(highlightthickness=0)
    buton_label_frame4.place_forget()


button_the_frame4.bind("<Enter>", on_enter_buton_f4)
button_the_frame4.bind("<Leave>", on_leave_buton_f4)


#--------------FRAME1 BUTTON------------
def go_frame2():
    show_frame(frame2)


def go_frame3():
    show_frame(frame3)


def go_frame4():
    show_frame(frame4)


text_label_info = tkinter.Label(frame1, text="Text", font=("Terminal", 25, "bold"), bg="black", fg="green2")


def on_enter_text(event):
    event.widget.config(highlightbackground="green2", highlightthickness=3)
    text_label_info.place(x=110, y=350)


def on_leave_text(event):
    event.widget.config(highlightthickness=0)
    text_label_info.place_forget()


button_text = tkinter.Button(frame1, image=photo_text, bg="white", borderwidth=0, relief="flat", command=go_frame2)
button_text.place(x=70, y=210)
button_text.bind("<Enter>", on_enter_text)
button_text.bind("<Leave>", on_leave_text)

web_label_info = tkinter.Label(frame1, text="Web", font=("Terminal", 25, "bold"), bg="black", fg="green2")


def on_enter_web(event):
    event.widget.config(highlightbackground="green2", highlightthickness=3)
    web_label_info.place(x=367, y=350)


def on_leave_web(event):
    event.widget.config(highlightthickness=0)
    web_label_info.place_forget()


button_web = tkinter.Button(frame1, image=photo_web, bg="black", borderwidth=0, relief="flat", command=go_frame3)
button_web.place(x=325, y=210)
button_web.bind("<Enter>", on_enter_web)
button_web.bind("<Leave>", on_leave_web)

doc_label_info = tkinter.Label(frame1, text="File", font=("Terminal", 25, "bold"), bg="black", fg="green2")


def on_enter_doc(event):
    event.widget.config(highlightbackground="green2", highlightthickness=3)
    doc_label_info.place(x=630, y=350)


def on_leave_doc(event):
    event.widget.config(highlightthickness=0)
    doc_label_info.place_forget()


button_doc = tkinter.Button(frame1, image=photo_doc, bg="white", borderwidth=0, relief="flat", command=go_frame4)
button_doc.place(x=580, y=210)
button_doc.bind("<Enter>", on_enter_doc)
button_doc.bind("<Leave>", on_leave_doc)

show_frame(frame1)
frame1.config(bg="black")
window.mainloop()