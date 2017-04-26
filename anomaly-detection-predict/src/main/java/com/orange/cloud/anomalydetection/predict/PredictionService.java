package com.orange.cloud.anomalydetection.predict;

import com.orange.cloud.anomalydetection.predict.domain.Observation;
import com.orange.cloud.anomalydetection.predict.domain.Prediction;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Sebastien Bortolussi
 */
public interface PredictionService {

    Prediction predict(Observation observation) throws Exception;

    INDArray predict(INDArray observation) throws Exception;
}
