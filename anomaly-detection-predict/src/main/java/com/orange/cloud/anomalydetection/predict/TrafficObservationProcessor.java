package com.orange.cloud.anomalydetection.predict;

import com.orange.cloud.anomalydetection.message.PredictionMessage;
import com.orange.cloud.anomalydetection.predict.domain.Observation;
import com.orange.cloud.anomalydetection.predict.domain.Prediction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.cloud.stream.annotation.StreamListener;
import org.springframework.cloud.stream.messaging.Processor;
import org.springframework.messaging.handler.annotation.SendTo;
import org.springframework.stereotype.Component;

/**
 * @author Sebastien Bortolussi
 */
@Component
public class TrafficObservationProcessor {

    private static final Logger LOGGER = LoggerFactory.getLogger(TrafficObservationProcessor.class);

    PredictionService predictionService;

    public TrafficObservationProcessor(PredictionService predictionService) {
        this.predictionService = predictionService;
    }

    @StreamListener(Processor.INPUT)
    @SendTo(Processor.OUTPUT)
    public PredictionMessage predict(String value) throws Exception {
        Observation observation = new Observation(new Double(value));
        final Prediction prediction = predictionService.predict(observation);
        LOGGER.debug("prediction : + " + prediction);
        return new PredictionMessage(String.valueOf(observation.getValue()), String.valueOf(prediction.getValue()));
    }
}
