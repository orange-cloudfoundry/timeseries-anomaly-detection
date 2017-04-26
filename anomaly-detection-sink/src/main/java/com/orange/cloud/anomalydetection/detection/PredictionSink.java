package com.orange.cloud.anomalydetection.detection;

import com.orange.cloud.anomalydetection.message.PredictionMessage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.cloud.stream.annotation.StreamListener;
import org.springframework.cloud.stream.messaging.Sink;
import org.springframework.stereotype.Component;

/**
 * @author Sebastien Bortolussi
 */
@Component
public class PredictionSink {

    private static final Logger LOGGER = LoggerFactory.getLogger(PredictionSink.class);

    @StreamListener(Sink.INPUT)
    public void predict(PredictionMessage value) throws Exception {
        LOGGER.debug("observation(t): " + value.getObservation() + "-> prediction(t+1): " + value.getPrediction());
    }
}
