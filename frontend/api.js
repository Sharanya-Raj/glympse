/**
 * Glyimpse Frontend API Client
 * Connects the HTML/JS frontend to the FastAPI backend (http://localhost:8000)
 */

const API_BASE_URL = window.GLYIMPSE_API_URL || 'http://localhost:8000';

class GlyimpseAPIClient {
  constructor(baseUrl = API_BASE_URL) {
    this.baseUrl = baseUrl;
    this.token = localStorage.getItem('glyimpse_token') || null;
  }

  setToken(token) {
    this.token = token;
    if (token) {
      localStorage.setItem('glyimpse_token', token);
    } else {
      localStorage.removeItem('glyimpse_token');
    }
  }

  getHeaders(includeAuth = true, contentType = 'application/json') {
    const headers = {};
    if (contentType) {
      headers['Content-Type'] = contentType;
    }
    if (includeAuth && this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }
    return headers;
  }

  async checkHealth() {
    try {
      const res = await fetch(`${this.baseUrl}/health`);
      return await res.json();
    } catch (err) {
      console.warn('[GlyimpseAPI] Health check failed:', err);
      return { status: 'offline', error: err.message };
    }
  }

  async signUp(username, password) {
    const res = await fetch(`${this.baseUrl}/accounts`, {
      method: 'POST',
      headers: this.getHeaders(false),
      body: JSON.stringify({ username, password })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Sign up failed');
    if (data.access_token) this.setToken(data.access_token);
    return data;
  }

  async signIn(username, password) {
    const res = await fetch(`${this.baseUrl}/sessions`, {
      method: 'POST',
      headers: this.getHeaders(false),
      body: JSON.stringify({ username, password })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Sign in failed');
    if (data.access_token) this.setToken(data.access_token);
    return data;
  }

  async uploadHistoryCSV(file) {
    const formData = new FormData();
    formData.append('file', file);

    const headers = {};
    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    const res = await fetch(`${this.baseUrl}/history`, {
      method: 'POST',
      headers,
      body: formData
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'CSV upload failed');
    return data;
  }

  async predictGlucoseMeal(meal, meds = []) {
    // Format input for backend MealRequest schema
    const now = new Date();
    const dateStr = now.toISOString().split('T')[0];
    const hours = Math.floor(meal.t / 60).toString().padStart(2, '0');
    const mins = (meal.t % 60).toString().padStart(2, '0');
    const mealTime = `${dateStr} ${hours}:${mins}:00`;

    const recentCgm = (meal.recentCgm && meal.recentCgm.length >= 4)
      ? meal.recentCgm
      : [
          { timestamp: `${dateStr} 16:30:00`, glucose: 105.0 },
          { timestamp: `${dateStr} 16:40:00`, glucose: 108.0 },
          { timestamp: `${dateStr} 16:50:00`, glucose: 110.0 },
          { timestamp: `${dateStr} 17:00:00`, glucose: 112.0 }
        ];

    const payload = {
      meal_time: mealTime,
      dietary_intake: `${meal.name} (Carbs: ${meal.c}g, Protein: ${meal.p || 0}g, Fat: ${meal.f || 0}g)`,
      recent_cgm: recentCgm
    };

    try {
      const res = await fetch(`${this.baseUrl}/predict`, {
        method: 'POST',
        headers: this.getHeaders(true),
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Backend prediction failed');
      return data;
    } catch (err) {
      console.warn('[GlyimpseAPI] Remote prediction failed, fallback to local model:', err.message);
      return null;
    }
  }
}

// Global API singleton instance
window.glyimpseAPI = new GlyimpseAPIClient();
