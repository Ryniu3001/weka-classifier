package put.cs.idss.ml.weka;

import org.w3c.dom.Attr;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import javax.smartcardio.ATR;
import java.util.Enumeration;
import java.util.concurrent.ExecutionException;

public class NaiveBayesClassifier extends Classifier {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 7550409893545527343L;

	/** number of classes */
	protected int numClasses;

	/** counts, means, standard deviations, priors..... */
	//protected double[.....
    protected double [] class_Priors;
    protected double [][][] mCounts;

	/** srednie dla numerycznych atrybutow. */
	protected double [][] m_Means;

	/** Odchylenie standardowe dla numerycznych atrybutow */
	protected double [][] m_Devs;

    /** Constant for normal distribution. */
    protected static double NORM_CONST = Math.sqrt(2 * Math.PI);

	public NaiveBayesClassifier() {
		// TODO Auto-generated constructor stub
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {
		
		numClasses = data.numClasses();
		
		// remove instances with missing class
		data.deleteWithMissingClass();

		/* 1. Initialize arrays of counts for nominal attributes, 
		 * means and std.devs. for numeric attributes,
		 * and a priori probabilities of the classes. */
		double [][][] counts = new double[data.numClasses()][data.numAttributes()][];
        this.class_Priors = new double[data.numClasses()];
        this.m_Means = new double[data.numClasses()][data.numAttributes()-1];
        this.m_Devs = new double[data.numClasses()][data.numAttributes() -1];

		for(int i = 0; i < data.numAttributes() - 1; i++) {
			Attribute attribute = data.attribute(i);
			for(int j = 0; j < data.numClasses(); j++) {
                Enumeration<Instance> enumaration = data.enumerateInstances();
                while (enumaration.hasMoreElements()){
                    Instance instance = enumaration.nextElement();
                    if (j == (int)instance.classValue()) {
                        if (attribute.isNominal()) {
                            if (counts[j][i] == null)
                                counts[j][i] = new double[attribute.numValues()];
                            counts[j][i][(int) instance.value(i)] += 1;     //ile razy wystapila dana wartosc atrybutu dla danej klasy
                        } else {
                            if (counts[j][i] == null)
                                counts[j][i] = new double[1];
                            this.m_Means[(int)instance.classValue()][i] += instance.value(attribute); //iniciujemy wartosc sredniej sumą wartosci atrubutu dla danej klasy
							counts[j][i][0]++;		//ile wartosci ma dany atrybut w klasie
                        }
                    }
                }
			}
		}
		
		// 2. compute counts and sums.
		for(int i = 0; i < data.numInstances(); i++) {
			Instance instance = data.instance(i);
            this.class_Priors[(int)instance.classValue()]++;		//tu tylko zliczamy ilosc wystapien danej klasy
		}
		
		// 3. Compute means.
		for (int i = 0; i < data.numAttributes() - 1; i++){
			Attribute attribute = data.attribute(i);
			if (attribute.isNumeric()){
				for (int j = 0; j < data.numClasses(); j++){
					this.m_Means[j][i] /= counts[j][i][0] /*this.class_Priors[j]*/;		//srednia
				}
			}
		}
		// 4. Compute standard deviations.
		for (int insIdx = 0; insIdx < data.numInstances(); insIdx++){
			Instance instance = data.instance(insIdx);
			//if (!instance.classIsMissing()){
				for (int i = 0; i < data.numAttributes()-1; i++){
					Attribute attribute = data.attribute(i);
			//		if (!instance.isMissing(attribute)){
						if (attribute.isNumeric()){
							this.m_Devs[(int)instance.classValue()][i] += Math.pow((instance.value(attribute)       //licznik - Suma (wartosc - srednia)^2
                                    - this.m_Means[(int)instance.classValue()][i]),2) /**  (this.m_Means[(int)instance.classValue()][i] - instance.value(attribute))*/;
						}
		//			}
				}
		//	}
		}

		for (int i = 0; i < data.numAttributes()-1; i++){
            Attribute attribute = data.attribute(i);
            if (attribute.isNumeric()){
                for (int j = 0; j < data.numClasses(); j++){
                    if (this.m_Devs[j][i] <= 0){
                        throw new Exception("attribute " + attribute.name() +": standard deviation is 0 for class " +
                                data.classAttribute().value(j));
                    } else {
                        this.m_Devs[j][i] /= counts[j][i][0];               //mianownik - liczba prob
                        this.m_Devs[j][i] = Math.sqrt(this.m_Devs[j][i]);
                    }
                }
            }
        }

		// 5. normalize counts and a priori probabilities
        double sum = 0;
        for (int i = 0; i < data.numAttributes() - 1; i++){
            Attribute attribute = data.attribute(i);
            if (attribute.isNominal()){
                for (int j = 0; j < data.numClasses(); j++){
                    sum = Utils.sum(counts[j][i]);                          //licznosc atrybutu w klasie (czyli tak naprawde ile razy grypa tak, ile razy nie)
                    for (int a = 0; a < attribute.numValues(); a++){
                        counts[j][i][a] = counts[j][i][a] / sum;            // pradopodobienstwo wartosci atrybutu pod warunkiem klasy
                    }
                }
            }

        }

        sum = Utils.sum(this.class_Priors);
        for (int j = 0; j < data.numClasses(); j++)
            this.class_Priors[j] = (this.class_Priors[j]) / sum;    //prawdopodobienstwo klasy

        this.mCounts = counts;
	}
	
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		double[] distribution = new double[numClasses]; //rozkład
		
		for (int j = 0; j < instance.numClasses(); j++){
            distribution[j] = 1;
            for (int i=0; i < instance.numAttributes() - 1; i++){
                Attribute attribute = instance.attribute(i);
                if (!instance.isMissing(attribute)){
                    if (attribute.isNominal()){
                        distribution[j] *= this.mCounts[j][i][(int)instance.value(attribute)];   //P(X|klasa)
                    } else {
                        distribution[j] *= normalDens(instance.value(attribute), this.m_Means[j][i], this.m_Devs[j][i]);
                    }
                }
            }
            distribution[j] *= this.class_Priors[j];  //P(X|klasa) * P(klasa)
        }
		
		// Remember to normalize probabilities!
        Utils.normalize(distribution);          ////(P(X|klasa) * P(klasa)) / P(X) czyli przez sume licznikow dla kazdej klasy
		
		return distribution;    
	}
	
	@Override
	public double classifyInstance(Instance instance) throws Exception {
		double classValue = 0.0;
		double max = Double.MIN_VALUE;
		double[] dist = distributionForInstance(instance);
		
		for(int i = 0; i < dist.length; i++) {
			if(dist[i] > max) {
				classValue = i;
				max = dist[i];
			}
		}
		
		return classValue;
	}

    /**
     * Density function of normal distribution.
     *
     * @param x the value to get the density for
     * @param mean the mean
     * @param stdDev the standard deviation
     * @return the density
     */
    protected double normalDens(double x, double mean, double stdDev) {

        double diff = x - mean;

        return (1 / (NORM_CONST * stdDev))
                * Math.exp(-(diff * diff / (2 * stdDev * stdDev)));
    }

}
