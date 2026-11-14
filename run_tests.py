"""Senaryo testleri. Sonuçları results.csv dosyasına yazar (Excel'de açılır)."""
import argparse
import csv

from agent import build_agent, run

UZUN_METIN = (
    "Yapay zekâ ajanları, bir hedefe ulaşmak için ortamlarını gözlemleyen, karar veren ve "
    "araçlar aracılığıyla eyleme geçen sistemlerdir. Büyük dil modelleri bu sistemlerde karar "
    "verici çekirdek görevi görür. Model, hangi aracın çağrılacağına karar verir; araç uygulama "
    "tarafından çalıştırılır ve sonuç modele geri verilir. Bu döngü görev tamamlanana kadar "
    "tekrarlanır. Böylece model, hesaplama ve güncel bilgi arama gibi tek başına zayıf kaldığı "
    "işlerde dış araçlardan yararlanabilir."
)

# beklenen: kullanılması gereken araç kümesi (None = serbest), anahtar: cevapta geçmesi gereken ifade
SENARYOLAR = [
    {"ad": "S1 Tek araç: hesaplama", "soru": "15 * 24 + 7 kaçtır?",
     "beklenen": {"hesapla"}, "anahtar": ["367"]},
    {"ad": "S2 Tek araç: Vikipedi", "soru": "Alan Turing kimdir? Vikipedi'den kısaca anlat.",
     "beklenen": {"wikipedia_ara"}, "anahtar": ["Turing"]},
    {"ad": "S3 Tek araç: özetleme", "soru": "Şu metni özetle: " + UZUN_METIN,
     "beklenen": {"metin_ozetle"}, "anahtar": []},
    {"ad": "S4 Araç gerekmez", "soru": "Merhaba, nasılsın?",
     "beklenen": set(), "anahtar": []},
    {"ad": "S5 Çok adımlı: Vikipedi + hesaplama",
     "soru": "Vikipedi'ye göre Türkiye'nin nüfusu kaçtır? Bu sayının %15'ini hesapla.",
     "beklenen": {"wikipedia_ara", "hesapla"}, "anahtar": []},
    {"ad": "S6 Çok adımlı: kişi + hesaplama",
     "soru": "Alan Turing hangi yıl doğdu? O yıla 100 ekle.",
     "beklenen": {"wikipedia_ara", "hesapla"}, "anahtar": ["2012"]},
    {"ad": "S7 Eksik girdi", "soru": "Bunu özetle.",
     "beklenen": set(), "anahtar": []},
    {"ad": "S8 Araç hatası", "soru": "100 bölü 0 kaç eder?",
     "beklenen": None, "anahtar": []},
    {"ad": "S9 Genel bilgi (araç serbest)", "soru": "Dünyanın en yüksek dağı hangisi?",
     "beklenen": None, "anahtar": ["Everest"]},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen3:4b"])
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--pdf-soru", default=None, help="PDF ile ilgili test sorusu")
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    senaryolar = list(SENARYOLAR)
    if args.pdf and args.pdf_soru:
        senaryolar.append({"ad": "S10 PDF araması", "soru": args.pdf_soru,
                           "beklenen": {"pdf_ara"}, "anahtar": []})

    satirlar = []
    for model in args.models:
        agent = build_agent(model, args.pdf)
        for s in senaryolar:
            try:
                r = run(agent, s["soru"])
            except Exception as e:
                r = {"answer": f"HATA: {e}", "tools": [], "trace": [],
                     "llm_calls": 0, "seconds": 0}
            kullanilan = set(r["tools"])
            arac_ok = "" if s["beklenen"] is None else (kullanilan == s["beklenen"])
            anahtar_ok = (all(k.lower() in r["answer"].lower() for k in s["anahtar"])
                          if s["anahtar"] else "")
            satirlar.append({
                "model": model, "senaryo": s["ad"], "soru": s["soru"][:80],
                "beklenen_araclar": "serbest" if s["beklenen"] is None else ", ".join(sorted(s["beklenen"])) or "yok",
                "kullanilan_araclar": ", ".join(r["tools"]) or "yok",
                "arac_dogru": arac_ok, "anahtar_dogru": anahtar_ok,
                "model_cagrisi": r["llm_calls"], "sure_sn": r["seconds"],
                "cevap": r["answer"], "puan_elle_0_2": "",
            })
            print(f"[{model}] {s['ad']}: araç={r['tools']} ({r['seconds']} sn)")

    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(satirlar[0]))
        w.writeheader()
        w.writerows(satirlar)
    print(f"\nYazıldı: {args.out}")


if __name__ == "__main__":
    main()
