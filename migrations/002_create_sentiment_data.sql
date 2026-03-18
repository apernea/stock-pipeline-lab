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
    ticker_sentiment_label VARCHAR(20),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_sentiment_symbol_date ON sentiment_data (symbol, published_at);

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_sentiment_updated_at
BEFORE UPDATE ON sentiment_data
FOR EACH ROW EXECUTE FUNCTION update_updated_at();
