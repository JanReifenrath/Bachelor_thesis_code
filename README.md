# Bachelor_thesis_code
Der in diesem Repository geteilte Code wurde im Rahmen einer Bachelorarbeit entwickelt. Mit diesem Code wurde ein 
ConvNeXt-Modell trainiert, das OSM-"lanes"- Datenlücken füllen kann.

Im Folgenden wird kurz erklärt, wie die Skripte genutzt werden können, um selbst für gewählte 
Regionen "lanes"-Datenlücken zu füllen.

## Reihenfolge der Skripte 

1. download_DOPs.py: zum automatischen herunterladen von Kacheln in einem bestimmten Bereich. Zum Anpassen der Region
müssen hauptsächlich die ersten zwei Zahlen der Loops zur Erstellung einer Tiles-Liste angepasst werden


2. combine_DOPs.py: kombiniert alle DOP-Kacheln in eine Datei


3. osm_dop_to_convnext_dataset_for_training_data.py: Erstellt Trainingsdaten für das Modell. Hierfür müssen vorerst 
die OSM-Straßendaten von Straßen mit lanes-Informationen heruntergeladen werden.
In Overpass Turbo kann hierbei der folgende Filter angewandt werden:
(highway=motorway or highway=trunk or highway=primary or highway=secondary or highway=tertiary or highway=residential or highway=service or highway=living_street or highway=trunk_link or highway=motorway_link or highway=primary_link or highway=secondary_link or highway=tertiary_link or highway=unclassified) and lanes=*


4. remove_black_images.py: aus verschiedenen Gründen werden vom 3. Skript einige größtenteils schwarze Bilder erstellt. 
Dieses Skipt löscht solche Bilder.


5. training.py: trainiert das Modell


6. (classify_and_assign_majority_vote.py: Wendet das Modell auf gesamten Datensatz der Straßen mit lanes-Info an. 
Dient primär zur genaueren Analyse)


7. osm_dop_to_convnext_dataset_for_final_classification.py: erstellt Bilder für den gesamten Straßendatensatz, die später 
genutzt werden, um Datenlücken zu füllen. Zum erstellen vom finalen Datensatz dann erneut classify_and_assign_majority_vote.py
auf den hiermit erstellten Datensatz anwenden.


8. train_further.py: kann genutzt werden, um für ein trainiertes Modell fine-tuning auf eine andere Region anzuwenden


Mit dem hier erklärten workflow können die Ergebnisse aus der Bachelorarbeit reproduziert werden.
Zur erstellung von Plots wurden noch weitere Skripts genutzt, die hier nicht hochgeladen wurden weil sie nicht für den 
Haupt-Workflow relevant sind.
Der final erstelle Datensatz befindet sich in data/.

Bei Fragen zur Anwendung der Skripts oder zu den Plot-Skripts stehe ich gerne zur Verfügung.
