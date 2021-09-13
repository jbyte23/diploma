# Analiza tematik in sentimenta slovenskih medijev z orodji za obdelavo naravnega jezika

Repozitorij vsebuje izvorno kodo uporabljeno v diplomskem delu.
Vsa koda je zapisana v obliki Google Colaboratory beležnic, zato priporočamo uporabo Google Colaboratory.

## Opis podatkov
V diplomskem delu smo uporabili podatke iz dveh različnih množic člankov: Event Registry (podatkovna zbirka ni javno dostopna) in SentiNews (https://www.clarin.si/repository/xmlui/handle/11356/1110).

V mapi `data` se nahajajo datoteke s podatki, ki jih uporabimo v diplomskem delu.
V diplomskem delu smo se omejili na članke sedmih slovenskih medijev (Dnevnik, 24ur.com, MMC RTV Slovenija, Siol.net Novice, Nova24TV, Tednik Demokracija, Portal Politikis) iz let 2019 in 2020.

Datoteke tipa `.pkl` predstavljajo podatke o slovenskih člankih. Vsaka datoteka ima informacije o naslovu članka, viru članka, število besed v članku, seznam predprocesiranih besed, id dodeljene teme in verjetnost pripadnosti dodeljeni temi.
Določene datoteke (`2019_slo_politika`, `2019_svet`, `2020_slo_politika`, `2020_svet`, `2020_korona`) vsebujejo še informacijo o zaznanem sentimentu (pozitivno, nevtralno, negativno).

Datoteka `preprocessed_data` vsebuje predprocesirane članke podatkovne množice SentiNews.



## Opis modelov LDA
V mapi `lda_models` se nahajata dve mapi, ena za vsako obravnavano leto. V vsaki mapi se nahajajo datoteke z modeli LDA za splošne in podrobne teme. V vsaki mapi so shranjeni tudi slovarji in korpus za vsak model. Več podrobnosti je opisanih v samih datotekah z izvorno kodo, kjer uporabimo te modele.


## Opis datotek s kodo
Vse datoteke s kodo odprite z Google Colaboratory.
Podrobnosti uporabe posamezne datoteke je opisan znotraj posameznih datotek.
Za pravilno uporabo datotek, je priporočljivo, da so vse datoteke naložene v Google Drive. V nasprotnem primeru pa je potrebno ustrezno spremeniti poti do datotek v izvorni kodi.

V datoteki `utils.py` so nekatere pomožne funkcije, ki jih uporabimo v ostalih datotekah.

### Procesiranje podatkov
V datoteki `data_processing.ipynb` preberemo članke iz podatkovne zbirke Event Registry in jih predprocesiramo za nadaljno uporabo.
Ker podatkovna množica Event Registry ni javno dostopna, datoteke ni možno zagnati. Njen namen je zgolj prikaz postopka predprocesiranja podatkov.

### Modeliranje tem
Procesirane podatke upo
Učenje modela LDA izvedemo v datoteki `topic_modeling.ipynb`, kjer zgradimo model LDA in ga shranimo. To datoteko uporabimo, če želimo naučiti nov model LDA in ga shraniti.

Interpretacija modela se izvede v datoteki `topic_interpretation.ipynb`, kjer opravimo interpretacijo tem modela LDA. To datoteko uporabimo, ko imamo shranjene LDA modele in jih želimo interpretirati.

### Zaznavanje sentimenta
V datoteki `sentiment_train.ipynb` je izvorna koda za prilagajanje modela SloBERTa za nalogo zaznavanja sentimenta. S to datoteko smo naučili model SloBERTa, ki ga uporabimo za zaznavanje sentimenta. Za uporabo te datoteke, je potrebno najprej shraniti še članke iz podatkovne množice SentiNews.

V datoteki `svm_train.ipynb` je izvorna koda za učenje SVM klasifikatorja za zaznavanje sentimenta.

V datoteki `sentiment_classification.ipynb` izvedemo klasifikacijo modela. Ta datoteka za delovanje potrebuje besedila člankov iz Event Registry podatkovne zbirke, ki pa niso javno dostopni. Sentiment za te članke je shranjen v določenih .pkl datotekah opisane zgoraj.

V datoteki `sentiment_analysis.ipynb` opravimo analizo zaznavanja sentimenta.
