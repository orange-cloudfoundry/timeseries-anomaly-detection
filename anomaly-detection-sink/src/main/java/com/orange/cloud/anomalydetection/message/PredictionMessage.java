package com.orange.cloud.anomalydetection.message;

/**
 * @author Sebastien Bortolussi
 */
public class PredictionMessage {

    private String observation;

    private String prediction;

    public PredictionMessage(String observation, String prediction) {
        this.observation = observation;
        this.prediction = prediction;
    }

    public String getObservation() {
        return observation;
    }

    public void setObservation(String observation) {
        this.observation = observation;
    }

    public String getPrediction() {
        return prediction;
    }

    public void setPrediction(String prediction) {
        this.prediction = prediction;
    }
}
