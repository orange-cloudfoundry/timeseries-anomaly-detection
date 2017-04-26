package com.orange.cloud.anomalydetection.detection.domain;

/**
 * @author Sebastien Bortolussi
 */
public class Prediction {

    private Double value;

    public Prediction(Double value) {
        this.value = value;
    }

    public Double getValue() {
        return value;
    }

    public void setValue(Double value) {
        this.value = value;
    }
}
