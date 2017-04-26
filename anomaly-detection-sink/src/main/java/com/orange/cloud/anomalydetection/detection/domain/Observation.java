package com.orange.cloud.anomalydetection.detection.domain;

/**
 * @author Sebastien Bortolussi
 */
public class Observation {

    private Double value;

    public Observation(Double value) {
        this.value = value;
    }

    public Double getValue() {
        return value;
    }

    public void setValue(Double value) {
        this.value = value;
    }
}
