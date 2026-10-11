/**
 * Glyimpse API client. Every failure surfaces as an ApiError with a user-facing
 * message; there is deliberately no local/offline prediction fallback.
 */
(function () {
  const TOKEN_KEY = 'glyimpse_cognito_access_token';
  localStorage.removeItem('glyimpse_token'); // Discard legacy SQLite sessions.

  class ApiError extends Error {
    constructor(kind, message, status) {
      super(message);
      this.kind = kind;     // offline | auth | forbidden | no_history | too_large | invalid | model_missing | server
      this.status = status; // HTTP status, 0 when the server could not be reached
    }
  }

  function detailText(detail) {
    if (Array.isArray(detail)) {
      // FastAPI validation errors: [{loc: [...], msg: '...'}]
      return detail.map(d => {
        const field = (d.loc || []).filter(p => p !== 'body').join('.');
        return field ? `${field}: ${d.msg}` : d.msg;
      }).join('; ');
    }
    return typeof detail === 'string' ? detail : '';
  }

  class GlyimpseAPIClient {
    constructor(baseUrl) {
      this.baseUrl = (baseUrl || '').replace(/\/+$/, '');
      this.token = localStorage.getItem(TOKEN_KEY);
      this.onUnauthorized = null;
    }

    get serverLabel() {
      return this.baseUrl || location.origin;
    }

    setToken(token) {
      this.token = token || null;
      if (token) localStorage.setItem(TOKEN_KEY, token);
      else localStorage.removeItem(TOKEN_KEY);
    }

    async request(method, path, { json, form, auth = true, signIn = false, signal } = {}) {
      const headers = {};
      if (json !== undefined) headers['Content-Type'] = 'application/json';
      if (auth && this.token) headers['Authorization'] = `Bearer ${this.token}`;
      let res;
      try {
        res = await fetch(this.baseUrl + path, {
          method, headers, signal,
          body: json !== undefined ? JSON.stringify(json) : form
        });
      } catch (err) {
        if (err.name === 'AbortError') throw err;
        throw new ApiError('offline',
          `Can't reach the Glyimpse server at ${this.serverLabel}. Check that the backend is running, then try again.`, 0);
      }
      if (res.status === 204) return null;
      let data = null;
      try { data = await res.json(); } catch (err) { /* non-JSON body */ }
      if (res.ok) return data;

      const detail = detailText(data && data.detail);
      switch (res.status) {
        case 401:
          if (signIn) throw new ApiError('auth', 'Wrong email or password.', 401);
          if (!auth) throw new ApiError('auth', detail || 'Unable to complete this request. Check your details and try again.', 401);
          this.setToken(null);
          if (this.onUnauthorized) this.onUnauthorized();
          throw new ApiError('auth', 'Your sign-in expired. Please sign in again.', 401);
        case 403:
          throw new ApiError('forbidden', detail || 'Not allowed.', 403);
        case 409:
          throw new ApiError(signIn ? 'auth' : (detail === 'Upload glucose history first' ? 'no_history' : 'conflict'), detail || 'Request conflicts with current data.', 409);
        case 413:
          throw new ApiError('too_large', detail || 'That file is too large (10 MB max).', 413);
        case 422:
          throw new ApiError('invalid', detail || 'The server could not use that input.', 422);
        case 503:
          if (path.startsWith('/food/')) throw new ApiError('unavailable', detail || 'Food service unavailable.', 503);
          throw new ApiError('unavailable', detail || 'Backend service unavailable. Please try again later.', 503);
        default:
          throw new ApiError('server', detail || `Server error (${res.status}). Try again in a moment.`, res.status);
      }
    }

    health() { return this.request('GET', '/health', { auth: false }); }

    async signIn(email, password) {
      const data = await this.request('POST', '/sessions', { json: { email, password }, auth: false, signIn: true });
      this.setToken(data.access_token);
      return data;
    }

    forgotPassword(email) {
      return this.request('POST', '/password/forgot', { json: { email }, auth: false });
    }

    async resetPassword(email, code, password) {
      const data = await this.request('POST', '/password/reset', { json: { email, code, password }, auth: false });
      this.setToken(null);
      return data;
    }

    async signOut() {
      try { if (this.token) await this.request('DELETE', '/sessions'); }
      finally { this.setToken(null); }
    }

    me() { return this.request('GET', '/me'); }
    getHistory() { return this.request('GET', '/history'); }
    getReadings() { return this.request('GET', '/history/readings'); }

    uploadHistory(file) {
      const form = new FormData();
      form.append('file', file);
      return this.request('POST', '/history', { form });
    }

    /**
     * meal: { meal_time?, dietary_intake, insulin?: {name, dose}, tablet?: {name, dose}, confirm? }.
     * meal_time defaults to the live clock's "now"; recent CGM comes from the server's history.
     * confirm=false is a what-if (nothing saved); confirm=true records that the meal was eaten.
     */
    predict(meal) { return this.request('POST', '/predict', { json: meal }); }

    /** Model forecasts for changed versions of the meal (skip sweets, smaller portions). Nothing is saved. */
    alternatives(meal) { return this.request('POST', '/predict/alternatives', { json: meal }); }

    /** Replay clock state plus readings/events up to now (only newer than `since` when given). */
    live(since) { return this.request('GET', '/live' + (since ? '?since=' + encodeURIComponent(since) : '')); }
    control(action, extra = {}) { return this.request('POST', '/live/control', { json: { action, ...extra } }); }
    meals() { return this.request('GET', '/meals'); }
    catalog() { return this.request('GET', '/catalog', { auth: false }); }

    dailyExplanations() { return this.request('GET', '/explanations/daily'); }
    modelAccuracy() { return this.request('GET', '/model/accuracy'); }

    // --- Photo/typed food -> USDA nutrients & standard portions ---
    /** Identify a food photo (FormData with an "image" field). -> { predictions, needs_confirmation }. */
    async uploadFoodImage(file) {
      if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type) || !file.size || file.size > 5 * 1024 * 1024) {
        throw new ApiError('invalid', 'Choose a JPEG, PNG or WebP image up to 5 MB.', 422);
      }
      for (let attempt = 0; attempt < 2; attempt++) {
        const intent = await this.request('POST', '/food/images/uploads', { json: { content_type: file.type, size_bytes: file.size } });
        const form = new FormData();
        Object.entries(intent.upload.fields).forEach(([key, value]) => form.append(key, value));
        form.append('file', file); // S3 requires the file field last. No Cognito header goes to S3.
        let uploaded;
        try { uploaded = await fetch(intent.upload.url, { method: 'POST', body: form, signal: AbortSignal.timeout(60000), credentials: 'omit' }); }
        catch (e) { throw new ApiError('offline', 'Photo upload failed or timed out. Select the photo again to retry.', 0); }
        if (uploaded.status === 403 && attempt === 0) continue; // Renew an expired upload policy once.
        if (!uploaded.ok) throw new ApiError('invalid', 'S3 rejected the photo upload. Select the photo again to retry.', uploaded.status);
        return this.request('POST', '/food/images/' + intent.image_id + '/complete');
      }
    }
    foodImageUrl(imageId) { return this.request('GET', '/food/images/' + encodeURIComponent(imageId)); }

    identifyFood(form) { return this.request('POST', '/food/identify', { form }); }
    dictateMeal(form, signal) { return this.request('POST', '/food/dictate', { form, signal }); }
    /** USDA nutrients + standard portions for a classifier label or typed food name. */
    foodNutrients(label) { return this.request('GET', '/food/nutrients?label=' + encodeURIComponent(label)); }
    /** USDA nutrients + standard portions for a specific fdcId (e.g. a "none of these" search pick). */
    foodNutrientsById(fdcId) { return this.request('GET', '/food/nutrients/' + encodeURIComponent(fdcId)); }
    /** Up to 10 USDA matches for free text (powers the "none of these, search" fallback). */
    foodSearch(q) { return this.request('GET', '/food/search?q=' + encodeURIComponent(q)); }
  }

  window.ApiError = ApiError;
  window.glyimpseAPI = new GlyimpseAPIClient(window.GLYIMPSE_API_URL);
})();
