# Using Diffusion Denoising Operators for Climate Data Analysis

## Data Setup

We use the [EUMSTAT data repository](https://data.eumetsat.int) which is a convenient way to access both remote sensory (sattellite) and in-situ climate data.
Make sure to create an account and store your credentials.
We use python-dotenv to store and load these credentials, which ensures they do not accidentally get committed to the repository!
The `.env` file should thus have the following:

```bash
# access https://api.eumetsat.int/api-key/ from your user profile
EUMESTAT_CONSUMER_KEY=<consumer-key>
EUMESTAT_CONSUMER_SECRET=<consumer-secret>
```
