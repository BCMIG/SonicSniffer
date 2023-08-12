const express = require('express');
const multer = require('multer');
const bodyParser = require('body-parser');
const cors = require('cors');
const fs = require('fs');

const app = express();
const port = 3010;

// Middleware
app.use(cors()); // Enable CORS for all routes
app.use(bodyParser.json());

// Set up multer for file storage
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/');
    },
    filename: (req, file, cb) => {
        const timestamp = Date.now();
        req.timestamp = timestamp; // Store the timestamp in the request object for later use
        cb(null, `${timestamp}.wav`);
    }
});

const upload = multer({ storage: storage });

app.post('/save', upload.single('audio'), (req, res) => {
    let keypresses = JSON.parse(req.body.keypresses);

    // Use the stored timestamp from the request object to name the .json file
    // pretty print json
    fs.writeFileSync(`uploads/${req.timestamp}.json`, JSON.stringify(keypresses, null, 4));

    res.send({ message: 'Successfully saved data!' });
});

app.listen(port, () => {
    console.log(`Server started on http://localhost:${port}`);
});
