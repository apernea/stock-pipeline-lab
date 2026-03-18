CREATE TABLE predictions (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    -- What was predicted, by whom, and when
    symbol          VARCHAR(10)      NOT NULL,
    model_name      VARCHAR(50)      NOT NULL,
    model_version   VARCHAR(20)      NOT NULL DEFAULT 'unknown',
    prediction_date DATE             NOT NULL,  -- date the prediction was made
    target_date     DATE             NOT NULL,  -- date being predicted
    horizon_days    SMALLINT         NOT NULL DEFAULT 1,

    -- Prediction outputs
    predicted_close  DOUBLE PRECISION NOT NULL,  -- absolute price target
    predicted_return DOUBLE PRECISION NOT NULL,  -- % change from last known close
    direction        SMALLINT         NOT NULL,  -- 1=up, -1=down, 0=neutral
    confidence       DOUBLE PRECISION,           -- model confidence score [0, 1]
    lower_bound      DOUBLE PRECISION,           -- prediction interval lower bound
    upper_bound      DOUBLE PRECISION,           -- prediction interval upper bound

    -- Actuals — filled in after target_date passes (by a scoring job)
    actual_close     DOUBLE PRECISION,
    actual_return    DOUBLE PRECISION,
    mae              DOUBLE PRECISION,           -- |predicted_close - actual_close|
    direction_correct BOOLEAN,                  -- did direction match?

    created_at      TIMESTAMP        NOT NULL DEFAULT NOW(),

    -- One prediction per symbol/model/version/date/horizon
    CONSTRAINT uq_prediction UNIQUE (symbol, model_name, model_version, prediction_date, target_date, horizon_days),

    CONSTRAINT ck_direction CHECK (direction IN (-1, 0, 1)),
    CONSTRAINT ck_confidence CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    CONSTRAINT ck_horizon CHECK (horizon_days > 0),
    CONSTRAINT ck_bounds CHECK (lower_bound IS NULL OR lower_bound <= upper_bound)
);

CREATE INDEX idx_predictions_symbol_target ON predictions (symbol, target_date DESC);
CREATE INDEX idx_predictions_model ON predictions (model_name, model_version);
CREATE INDEX idx_predictions_unscored ON predictions (target_date) WHERE actual_close IS NULL;

-- Trigger to auto-compute mae and direction_correct when actuals are filled in
CREATE OR REPLACE FUNCTION fill_prediction_metrics()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.actual_close IS NOT NULL AND NEW.predicted_close IS NOT NULL THEN
        NEW.mae := ABS(NEW.predicted_close - NEW.actual_close);
    END IF;
    IF NEW.actual_return IS NOT NULL AND NEW.predicted_return IS NOT NULL THEN
        NEW.direction_correct := (SIGN(NEW.predicted_return) = SIGN(NEW.actual_return));
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_fill_prediction_metrics
BEFORE INSERT OR UPDATE ON predictions
FOR EACH ROW EXECUTE FUNCTION fill_prediction_metrics();
