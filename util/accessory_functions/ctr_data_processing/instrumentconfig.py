'''
Instrument Configuration
Author: John Hammonds (JPHammonds@anl.gov)
1/11/2017
'''
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError


IC_XMLNS = "https://github.com/xraypy/surface_integrator/instConfig"
ANGLE_LABELS = "angle_labels"
PANGLE_LABELS = "pangle_labels"
SCALER_LABELS = "scaler_labels"
# angle labels
CHI = "chi"
DEL = "del"
ETA = "eta"
MU = "mu"
NU = "nu"
PHI = "phi"
#Pseudo angle Labels
P_CHI = "chi"
P_DEL = "TwoTheta"
P_ETA = "theta"
P_MU = "Psi"
P_NU = "Nu"
P_PHI = "phi"
# Scaler labels
IO = "io"
TRANSM = "transm"
SECONDS = "Seconds"
CORRDET = "corrdet"
FILTERS = "filters"

class InstrumentConfiguration():
    def __init__(self, configFileName):
        self.configFileName = configFileName
        self.root = None
        if self.configFileName != None:
            try:
                self.tree = ET.parse(configFileName)
                self.root = self.tree.getroot()
                print ET.dump(self.root)
            except ParseError as pe:
                self.root = None
                print pe
                raise pe
        
    def readAngleLabels(self):
        angleLabels = []
        if (self.configFileName != None) and (self.root != None):
            labelsName = self.packNameSpaceName((ANGLE_LABELS,))
            #print labelsName
            angleLabelsE = self.root.findall(labelsName)[0]
            chiName = self.packNameSpaceName((CHI,))
            angleLabels.append(angleLabelsE.findall(chiName)[0].text)
            delName = self.packNameSpaceName((DEL,))
            angleLabels.append(angleLabelsE.findall(delName)[0].text)
            etaName = self.packNameSpaceName((ETA,))
            angleLabels.append(angleLabelsE.findall(etaName)[0].text)
            muName = self.packNameSpaceName((MU,))
            angleLabels.append(angleLabelsE.findall(muName)[0].text)
            nuName = self.packNameSpaceName((NU,))
            angleLabels.append(angleLabelsE.findall(nuName)[0].text)
            phiName = self.packNameSpaceName((PHI,))
            angleLabels.append(angleLabelsE.findall(phiName)[0].text)
        else:
            angleLabels = [CHI, DEL, ETA, MU, NU, PHI]
        return angleLabels
         
    def readDefScalerLabels(self):
        return [IO, TRANSM, SECONDS, CORRDET, FILTERS]
    
    def readPseudoAngleLabels(self):
        angleLabels = []
        
        if (self.configFileName != None) and (self.root != None):
            labelsName = self.packNameSpaceName((PANGLE_LABELS,))
            angleLabelsE = self.root.findall(labelsName)[0]
            chiName = self.packNameSpaceName((P_CHI,))
            angleLabels.append(angleLabelsE.findall(chiName)[0].text)
            delName = self.packNameSpaceName((P_DEL,))
            angleLabels.append(angleLabelsE.findall(delName)[0].text)
            etaName = self.packNameSpaceName((P_ETA,))
            angleLabels.append(angleLabelsE.findall(etaName)[0].text)
            muName = self.packNameSpaceName((P_MU,))
            angleLabels.append(angleLabelsE.findall(muName)[0].text)
            nuName = self.packNameSpaceName((P_NU,))
            angleLabels.append(angleLabelsE.findall(nuName)[0].text)
            phiName = self.packNameSpaceName((P_PHI,))
            angleLabels.append(angleLabelsE.findall(phiName)[0].text)
        else:
            angleLabels = [P_CHI, P_DEL, P_ETA, P_MU, P_NU, P_PHI]
        return angleLabels
         
    def readScalerLabels(self):
        scalerLabels = []
        if (self.configFileName != None) and (self.root != None):
            labelsName = self.packNameSpaceName((SCALER_LABELS,))
            scalerLabelsE = self.root.findall(labelsName)[0]
            ioName = self.packNameSpaceName((IO,))
            scalerLabels.append(scalerLabelsE.findall(ioName)[0].text)
            transmName = self.packNameSpaceName((TRANSM,))
            scalerLabels.append(scalerLabelsE.findall(transmName)[0].text)
            secondsName = self.packNameSpaceName((SECONDS,))
            scalerLabels.append(scalerLabelsE.findall(secondsName)[0].text)
            corrdetName = self.packNameSpaceName((CORRDET,))
            scalerLabels.append(scalerLabelsE.findall(corrdetName)[0].text)
            filtersName = self.packNameSpaceName((FILTERS,))
            scalerLabels.append(scalerLabelsE.findall(filtersName)[0].text)
        else:
            scalerLabels = self.readDefScalerLabels()
        return scalerLabels
         

    def packNameSpaceName(self, tags):
        ns = '{' + IC_XMLNS + '}'
        s = '/'
        nsName = ns + s.join(tags)
        #print nsName
        return nsName
        
if __name__ == '__main__':
    testFileName = "./resources/instConfig_13ID.xml"
    ic = InstrumentConfiguration(testFileName)
    ic.packNameSpaceName((ANGLE_LABELS, CHI))
    ic.packNameSpaceName((ANGLE_LABELS, DEL))
    angleLabels = ic.readAngleLabels()
    print angleLabels
    pseudoAngleLabels = ic.readPseudoAngleLabels()
    print pseudoAngleLabels
    scalerLabels = ic.readScalerLabels()
    print scalerLabels
    
    ic2 = InstrumentConfiguration(None)
    angleLabels = ic2.readAngleLabels()
    print angleLabels
    pseudoAngleLabels = ic2.readPseudoAngleLabels()
    print pseudoAngleLabels
    scalerLabels = ic2.readScalerLabels()
    print scalerLabels
    
    