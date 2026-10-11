/**
 * Where the Glyimpse API lives.
 *  ''                          -> same origin as this page (FastAPI serves the frontend locally)
 *  'http://127.0.0.1:8000'     -> a separately running local API
 *  'https://<api>.azurewebsites.net' -> Azure (set this when deploying the static site)
 * A value set before this file loads (e.g. an inline script) wins.
 */
window.GLYIMPSE_API_URL = "https://jte2702gol.execute-api.us-west-2.amazonaws.com/";
(location.protocol === 'file:' ? 'http://127.0.0.1:8000' : '');
