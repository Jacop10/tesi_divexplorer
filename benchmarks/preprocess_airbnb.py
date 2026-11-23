"""Preprocessing di Airbnb New York (Inside Airbnb, settembre 2024), fatto da zero.

Dati: Inside Airbnb (https://insideairbnb.com/get-the-data/), New York City, raccolta
del 4-5 settembre 2024, file listings.csv.gz dettagliato (37.541 annunci, 75 colonne).
E' lo stesso file usato nel repository di DivExplorer generalizzato; per scaricarlo:
  mkdir -p datasets/raw
  curl -L "https://raw.githubusercontent.com/elianap/divexplorer-generalized/main/data/listings.csv.gz" -o datasets/raw/listings.csv.gz
Uso (dalla cartella principale del repository):
  python3 benchmarks/preprocess_airbnb.py
Scrive datasets/airbnb_v2.csv (attributi discretizzati + outcome) e stampa un
riepilogo di ogni passo.

Outcome: review_scores_rating, il voto medio delle recensioni (numerico, 0-5).

Scelte (motivate nel riepilogo stampato):
1. righe: annunci con voto e con ALMENO 5 RECENSIONI. Con 1-2 recensioni il voto e'
   quasi casuale (deviazione standard 0,88 e 0,56, contro 0,33 con 5-9 recensioni e
   0,16 con 50 o piu'): fino a 5 recensioni il rumore cala molto, dopo poco.
   Tutti gli annunci, attivi e non: l'attivita' diventa un attributo;
2. attributi derivati:
   - attivo: almeno una recensione negli ultimi 12 mesi prima della data di raccolta;
   - visibile: l'annuncio era online alla raccolta (source = "city scrape");
     altrimenti Inside Airbnb riporta i dati di una raccolta precedente;
   - anni_host: anni dall'iscrizione dell'host; anni_annuncio: anni dalla prima
     recensione;
   - licenza: nessuna / esente / registrata (registro degli affitti brevi di NYC);
   - bagno_condiviso: si / no (dal testo del bagno, es. "1 shared bath");
   - servizi (amenities): un attributo si/no per ogni servizio presente in una
     quota di annunci tra il 20% e l'80% (quelli quasi universali, come il wifi, o
     rarissimi non distinguono gli annunci);
3. colonne tolte:
   - i sottovoti (review_scores_accuracy, _cleanliness, ...) e host_is_superhost:
     sono calcolati dal voto stesso (per essere superhost serve un voto medio di
     almeno 4,8), quindi spiegherebbero il voto in modo banale (leakage);
   - identificativi, URL, foto, testi liberi, nomi;
   - latitudine e longitudine (la posizione e' gia' in quartiere e distretto);
   - colonne quasi duplicate (varianti di notti minime/massime, di disponibilita'
     a 60/90 giorni, di conteggi degli annunci dell'host, recensioni negli ultimi 30
     giorni, quartiere dell'host);
4. numerici in 3 fasce con lo stesso numero di annunci (il prezzo in 4); se molti
   valori sono uguali le fasce possono essere meno di 3;
5. valori mancanti -> categoria "NA" (es. tasso di risposta mancante = l'host non ha
   mai ricevuto messaggi a cui rispondere); valori presenti in meno dell'1% degli
   annunci -> "other"; attributi quasi costanti (un valore in almeno il 99,5% degli
   annunci) tolti. Sono le stesse regole di Diabetes e HMDA.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "datasets", "raw", "listings.csv.gz")
OUT = os.path.join(ROOT, "datasets", "airbnb_v2.csv")

OUTCOME = "review_scores_rating"
MIN_REVIEWS = 5
AMENITY_RANGE = (0.20, 0.80)
RARE = 0.01
NEAR_CONSTANT = 0.995

CATEGORICAL = {   # colonna originale: nome dell'attributo
    "neighbourhood_group_cleansed": "distretto",
    "neighbourhood_cleansed": "quartiere",
    "room_type": "tipo_stanza",
    "property_type": "tipo_alloggio",
    "host_response_time": "tempo_risposta_host",
    "host_verifications": "verifiche_host",
    "host_has_profile_pic": "foto_profilo_host",
    "host_identity_verified": "identita_host_verificata",
    "instant_bookable": "prenotazione_immediata",
    "has_availability": "disponibile",
}
NUMERIC = {       # colonna originale: (nome dell'attributo, numero di fasce)
    "price": ("prezzo", 4),
    "accommodates": ("ospiti", 3),
    "bedrooms": ("camere", 3),
    "beds": ("letti", 3),
    "bathrooms": ("bagni", 3),
    "minimum_nights": ("notti_minime", 3),
    "maximum_nights": ("notti_massime", 3),
    "availability_30": ("disponibilita_30g", 3),
    "availability_365": ("disponibilita_365g", 3),
    "number_of_reviews": ("n_recensioni", 3),
    "number_of_reviews_ltm": ("n_recensioni_12mesi", 3),
    "reviews_per_month": ("recensioni_al_mese", 3),
    "host_response_rate": ("tasso_risposta_host", 3),
    "host_acceptance_rate": ("tasso_accettazione_host", 3),
    "calculated_host_listings_count": ("annunci_host", 3),
    "anni_host": ("anni_host", 3),
    "anni_annuncio": ("anni_annuncio", 3),
}


def _number(s):
    """'$1,234.00' -> 1234.0, '95%' -> 95.0, altrimenti numero; NaN se mancante."""
    return pd.to_numeric(s.astype(str).str.replace(r"[$,%]", "", regex=True),
                         errors="coerce")


def _bins(x, k):
    """k fasce con lo stesso numero di annunci; NaN -> "NA". Se un taglio cade sul
    valore massimo (molti annunci con lo stesso valore, es. tasso di risposta 100%),
    il massimo diventa una fascia a se', invece di finire con tutti gli altri."""
    out = pd.Series("NA", index=x.index, dtype=object)
    ok = x.notna()
    q = np.unique(np.quantile(x[ok], np.linspace(0, 1, k + 1)[1:-1]))
    top = x[ok].max()
    if len(q) and q[-1] >= top:
        below = x[ok][x[ok] < top]
        q = np.unique(np.append(q[q < top], below.max())) if len(below) else q[q < top]
    edges = [-np.inf] + [float(v) for v in q] + [np.inf]
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        labels.append(f"<={hi:g}" if i == 0 else (f">{lo:g}" if not np.isfinite(hi)
                                                    else f"({lo:g}-{hi:g}]"))
    out[ok] = pd.cut(x[ok], edges, labels=labels).astype(str)
    return out


def _license(s):
    s = s.fillna("").astype(str).str.strip().str.lower()
    return np.where(s.isin(["nan", "", "none"]), "nessuna",
                    np.where(s.str.contains("exempt"), "esente", "registrata"))


def preprocess(raw_path=RAW, out_path=OUT, verbose=True):
    def say(*a):
        if verbose:
            print(*a)

    raw = pd.read_csv(raw_path, low_memory=False)
    ref = pd.to_datetime(raw["last_scraped"]).max()
    say(f"File: {raw_path}\n  {len(raw)} annunci, {raw.shape[1]} colonne, "
        f"raccolta del {ref.date()}")

    # 1. righe
    df = raw[raw[OUTCOME].notna()]
    say(f"1. con voto: {len(df)} (tolti {len(raw) - len(df)} senza recensioni)")
    n0 = len(df)
    df = df[df["number_of_reviews"] >= MIN_REVIEWS].reset_index(drop=True)
    say(f"   con almeno {MIN_REVIEWS} recensioni: {len(df)} (tolti {n0 - len(df)})")

    # 2. attributi derivati
    last = pd.to_datetime(df["last_review"])
    df["attivo"] = np.where(last >= ref - pd.Timedelta(days=365), "si", "no")
    df["visibile"] = np.where(df["source"] == "city scrape", "si", "no")
    df["anni_host"] = ((ref - pd.to_datetime(df["host_since"])).dt.days / 365.25).round(1)
    df["anni_annuncio"] = ((ref - pd.to_datetime(df["first_review"])).dt.days
                           / 365.25).round(1)
    # verifiche dell'host: "['email', 'phone']" -> "email+phone"
    df["host_verifications"] = df["host_verifications"].map(
        lambda v: "+".join(json.loads(v.replace("'", '"'))) or "nessuna"
        if isinstance(v, str) and v.startswith("[") else v)
    df["licenza"] = _license(df["license"])
    bt = df["bathrooms_text"].astype(str).str.lower()
    df["bagno_condiviso"] = np.where(df["bathrooms_text"].isna(), "NA",
                                     np.where(bt.str.contains("shared"), "si", "no"))
    say(f"2. attivi (recensione negli ultimi 12 mesi): {np.mean(df['attivo'] == 'si'):.1%}; "
        f"visibili alla raccolta: {np.mean(df['visibile'] == 'si'):.1%}")

    out = pd.DataFrame(index=df.index)
    for c, name in CATEGORICAL.items():
        v = df[c].astype(object).where(df[c].notna(), "NA").astype(str)
        out[name] = v.replace({"t": "si", "f": "no"})             # booleani di Airbnb
    for name in ("attivo", "visibile", "licenza", "bagno_condiviso"):
        out[name] = df[name].astype(str)
    for c, (name, k) in NUMERIC.items():
        out[name] = _bins(_number(df[c]), k)

    # servizi
    amen = df["amenities"].map(lambda s: set(json.loads(s)) if isinstance(s, str) else set())
    counts = pd.Series([a for A in amen for a in A]).value_counts() / len(df)
    chosen = counts[(counts >= AMENITY_RANGE[0]) & (counts <= AMENITY_RANGE[1])].index
    for a in chosen:
        col = "servizio_" + "".join(ch if ch.isalnum() else "_" for ch in a.lower()).strip("_")
        out[col] = np.where(amen.map(lambda A, a=a: a in A), "si", "no")
    say(f"   servizi: {len(counts)} diversi, {len(chosen)} presenti nel "
        f"{AMENITY_RANGE[0]:.0%}-{AMENITY_RANGE[1]:.0%} degli annunci -> attributi si/no")

    # 5. valori rari e attributi quasi costanti
    for c in out.columns:
        f = out[c].value_counts(normalize=True)
        out[c] = out[c].where(~out[c].isin(f[f < RARE].index), "other")
    const = [c for c in out.columns
             if out[c].value_counts(normalize=True).iloc[0] >= NEAR_CONSTANT]
    out = out.drop(columns=const)
    say(f"5. tolti perche' quasi costanti: {const if const else 'nessuno'}")

    out[OUTCOME] = df[OUTCOME].astype(float).to_numpy()
    out.to_csv(out_path, index=False)
    y = out[OUTCOME]
    say(f"\nRisultato: {len(out)} annunci, {out.shape[1] - 1} attributi, "
        f"voto medio {y.mean():.3f} (dev. std {y.std():.3f})")
    say(f"Salvato in {out_path}\n\nAttributi (valori e quota di annunci):")
    for c in out.columns[:-1]:
        f = out[c].value_counts(normalize=True)
        say(f"  {c:32s} " + ", ".join(f"{v} {p:.0%}" for v, p in f.items()))
    return out


if __name__ == "__main__":
    preprocess()
