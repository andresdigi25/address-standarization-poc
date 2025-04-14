const express = require('express');
const NodeGeocoder = require('node-geocoder');

const app = express();
const port = 3000;

app.use(express.json());

const geocoder = NodeGeocoder({
  provider: 'openstreetmap'
});

app.post('/standardize-address', async (req, res) => {
  const { address, city, zipCode } = req.body;

  try {
    const [result] = await geocoder.geocode({ address, city, zipcode: zipCode, country: 'USA' });

    if (result) {
      res.json({
        standardizedAddress: result.formattedAddress,
        latitude: result.latitude,
        longitude: result.longitude
      });
    } else {
      res.status(404).json({ error: 'Address not found' });
    }
  } catch (error) {
    res.status(500).json({ error: 'An error occurred while standardizing the address' });
  }
});

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
