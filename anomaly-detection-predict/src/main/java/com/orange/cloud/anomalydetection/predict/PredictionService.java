package com.orange.cloud.anomalydetection.predict;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Sebastien Bortolussi
 */
public interface PredictionService {
    INDArray predict(INDArray featureMatrix) throws Exception;
}
