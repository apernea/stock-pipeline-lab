CREATE TABLE sentiment_data(
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    published_at TIMESTAMP NOT NULL,
    title TEXT NOT NULL,
    source VARCHAR(100),
    overall_sentiment_score DOUBLE PRECISION,
    overall_sentiment_label VARCHAR(20),
    ticker_relevance_score DOUBLE PRECISION,
    ticker_sentiment_score DOUBLE PRECISION,
    ticker_sentiment_label VARCHAR(20)
);

CREATE INDEX idx_sentiment_symbol_date ON sentiment_data (symbol, published_at);
