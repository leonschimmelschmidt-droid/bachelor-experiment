const express = require('express');
const fs = require('fs');
const path = require('path');
const cors = require('cors');

const app = express();
// NUTZT DEN PORT VON RENDER ODER LOKAL 3000
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static(__dirname));

const csvFilePath = path.join(__dirname, 'ergebnisse.csv');

if (!fs.existsSync(csvFilePath)) {
    const header = 'Zeitstempel;Treatment;Entscheidung_1;Entscheidung_2\n';
    fs.writeFileSync(csvFilePath, header, 'utf-8');
}

// Endpoint für den Datenempfang von der Webseite
app.post('/api/save', (req, res) => {
    const { treatment, decision1, decision2 } = req.body;
    const timestamp = new Date().toISOString();
    const csvLine = `${timestamp};${treatment};${decision1};${decision2}\n`;

    fs.appendFile(csvFilePath, csvLine, 'utf-8', (err) => {
        if (err) {
            console.error('Fehler beim Speichern:', err);
            return res.status(500).json({ status: 'error', message: 'Speichern fehlgeschlagen' });
        }
        console.log(`[Erfolg] Neue Daten gespeichert: Treatment ${treatment}`);
        res.json({ status: 'success', message: 'Daten erfolgreich gespeichert' });
    });
});

// GEHEIMER DOWNLOAD-LINK FÜR DEINE EXCEL/CSV-AUSWERTUNG
app.get('/download-ergebnisse-geheim', (req, res) => {
    if (fs.existsSync(csvFilePath)) {
        res.download(csvFilePath, 'bachelor_ergebnisse.csv');
    } else {
        res.status(404).send('Bisher wurden noch keine Daten eingetragen.');
    }
});

app.listen(PORT, () => {
    console.log(`==================================================`);
    console.log(` Server läuft auf Port ${PORT}`);
    console.log(`==================================================`);
});
