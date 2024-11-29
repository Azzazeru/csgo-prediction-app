const express = require('express');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');
const path = require('path');
const app = express();
const port = 8000;

app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

app.use(express.static(path.join(__dirname, 'public')));

app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());

app.get('/', (req, res) => {
    res.render('index');
});

app.post('/predict', (req, res) => {
    const data = req.body;
    const { RoundHeadshots, RoundFlankKills, PrimaryPistol, RoundAssists, PrimarySniperRifle, MatchKills } = data;

    if (
        RoundHeadshots < 0 || RoundHeadshots > 50 ||
        RoundFlankKills < 0 || RoundFlankKills > 20 ||
        PrimaryPistol < 0 || PrimaryPistol > 50 ||
        RoundAssists < 0 || RoundAssists > 20 ||
        PrimarySniperRifle < 0 || PrimarySniperRifle > 50 ||
        MatchKills < 0 || MatchKills > 500
    ) {
        return res.send('Error: Los valores de entrada no son válidos.');
    }

    const input = [
        parseFloat(RoundHeadshots),
        parseFloat(RoundFlankKills),
        parseFloat(PrimaryPistol),
        parseFloat(RoundAssists),
        parseFloat(PrimarySniperRifle),
        parseFloat(MatchKills)
    ];

    const python = spawn('python3', [path.join(__dirname, 'predict_model.py'), ...input]);

    let prediction = '';

    python.stdout.on('data', (data) => {
        prediction += data.toString();
    });

    python.stderr.on('data', (data) => {
        console.error(`Error de Python: ${data}`);
    });

    python.on('close', (code) => {
        if (code === 0) {
            res.render('result', { prediction: prediction });
        } else {
            res.send('Hubo un error al hacer la predicción.');
        }
    });
});

app.listen(port, () => {
    console.log(`Servidor corriendo en http://localhost:${port}`);
});
