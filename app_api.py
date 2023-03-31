
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from xgboost_preprocessing import api_call
from fastapi.responses import HTMLResponse

app = FastAPI()


class InputData(BaseModel):
    product: str
    dates: List[str]
    sales: List[float]
    ahead: int


@app.get("/", response_class=HTMLResponse)
async def read_items():
    html_content = """
    <!DOCTYPE html>
    <html>

    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Teeny-Tiny API 🏰</title>
    </head>

    <body>
    <h1 id="teeny-tiny-api-🏰">Teeny-Tiny API 🏰</h1>
    <p>This is an example of an ML model (<code>XGBRegressor</code>), with a particular pre-processing technique and configuration, that can be used for regression/forecasting tasks, served as an API.</p>
    <h2 id="how-to-send-requests">How to send <code>requests</code></h2>
    <pre class=" language-python"><code class="prism  language-python">
    <span class="token keyword">import</span> requests
    <span class="token keyword">import</span> pandas <span class="token keyword">as</span> pd

    <span class="token comment"># The data you are going to use</span>
    df <span class="token operator">=</span> pd<span class="token punctuation">.</span>read_csv<span class="token punctuation">(</span><span class="token string">'time_series_data.csv'</span><span class="token punctuation">)</span>

    <span class="token comment"># The API endpoint URL</span>
    url <span class="token operator">=</span> <span class="token string">"https://teeny-tiny-api.onrender.com/predict"</span>

    <span class="token comment"># Define the input data</span>
    data <span class="token operator">=</span> <span class="token punctuation">{</span>
        <span class="token string">"product"</span><span class="token punctuation">:</span> df<span class="token punctuation">.</span>product_id<span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">,</span> <span class="token comment"># name of the product</span>
        <span class="token string">"dates"</span><span class="token punctuation">:</span> <span class="token builtin">list</span><span class="token punctuation">(</span>df<span class="token punctuation">.</span>dates<span class="token punctuation">)</span><span class="token punctuation">,</span> <span class="token comment"># the list of dates</span>
        <span class="token string">"sales"</span><span class="token punctuation">:</span> <span class="token builtin">list</span><span class="token punctuation">(</span>df<span class="token punctuation">.</span>sales<span class="token punctuation">)</span><span class="token punctuation">,</span> <span class="token comment"># the list of sales</span>
        <span class="token string">"ahead"</span><span class="token punctuation">:</span> <span class="token number">15</span> <span class="token comment"># how many days ahead we want to look</span>
    <span class="token punctuation">}</span>

    <span class="token comment"># Send a POST request with the input data</span>
    response <span class="token operator">=</span> requests<span class="token punctuation">.</span>post<span class="token punctuation">(</span>url<span class="token punctuation">,</span> json<span class="token operator">=</span>data<span class="token punctuation">)</span>

    <span class="token comment"># Done!</span>
    <span class="token keyword">print</span><span class="token punctuation">(</span>response<span class="token punctuation">.</span>json<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span>

    </code></pre>
    <p>Return to the <a href="https://github.com/Nkluge-correa/teeny-tiny_castle">castle</a>.</p>

    </body>

    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.post("/predict")
async def predict(data: InputData):
    days, sales, statistics = api_call(data.dates, data.sales, data.ahead)
    return {data.product:
            {"dates": days,
             "sales": sales,
             "statistics": statistics}}

# uvicorn app_api:app --host 0.0.0.0 --port 10000 (render)
