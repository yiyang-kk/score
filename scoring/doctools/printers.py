from IPython.core.display import display, HTML
from .calculators import *

class Chapter(object):
    name = None
    children = []
    parent = None
    calculators = []
    
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.calculators = []
        if self.parent != None:
            #print('adding ref to parent '+parent.name)
            parent.add_subchapter(self)
    
    def add_subchapter(self, c):
        self.children.append(c)
    
    def get_subchapters(self):
        return self.children
    
    def add_calculator(self, cal):
        self.calculators.append(cal)
        
    def get_calculators(self):
        return self.calculators

    def get_hash(self):
        import hashlib
        hash_object = hashlib.md5(self.name.encode('utf-8'))
        return hash_object.hexdigest()



class StandardExecutionPlan(object):
    
    pp = None
    root = None
       
    def __init__(self, projectParameters):
        self.pp = projectParameters
        

    def calculate(self):
        pp = self.pp
        cont_pred = pp.predictors_continuous
        grp_pred = pp.predictors_grouped


        # Documentation root
        ch0 = Chapter('Scorecard documentation')
        self.root = ch0

        # Describe all samples
        ch10 = Chapter('Sample descriptions', ch0)

        for sample in pp.sample_ordering:
            sdc = SampleDescriptionCalculator(pp)
            sdc = sdc.s([(pp.sample_dict[sample], sample)])
            Chapter(sdc.get_description(), ch10).add_calculator(sdc)

        # Feature evaluation
        ch20 = Chapter('Predictor evaluation', ch0)

        # Binned features evaluation             
        ch22 = Chapter('Binned predictors', ch20)        

        for feature in grp_pred:
            ch = Chapter(feature, ch22)
            for target in pp.targets: 
                cht = Chapter(target[0], ch) 
                for sample in pp.sample_ordering:            
                    gec = GroupingEvaluationCalculator(pp)
                    gec = gec.s([(pp.sample_dict[sample], sample)]).p([feature]).t([target])
                    Chapter(gec.get_description(), cht).add_calculator(gec)


        # Continuous features evaluation
        ch21 = Chapter('Continuous predictors', ch20)

        for feature in cont_pred:
            ch = Chapter(feature, ch21)
            for target in pp.targets:
                cht = Chapter(target[0], ch) 
                for sample in pp.sample_ordering:
                    pst = ContinuousEvaluationCalculator(pp)    
                    pst = pst.s([(pp.sample_dict[sample], sample)]).p([feature]).t([target])
                    Chapter(pst.get_description(), cht).add_calculator(pst)



        # Evaluation of the score
        ch30 = Chapter('Score evaluation', ch0)

        ch31 = Chapter('Comparison with existing model(s)', ch30)
        for target in pp.targets:
            for sample in pp.sample_ordering:
                pgt = ScoreComparisonCalculator(pp)
                pgt = pgt.s([(pp.sample_dict[sample], sample)]).p(pp.scores).t([target])
                Chapter(pgt.get_description(), ch31).add_calculator(pgt)

        ch32 = Chapter('Evaluation of new model(s)', ch30)
        for score in [pp.scores[0]]:
            for target in pp.targets:
                for sample in pp.sample_ordering:
                    pgt = ContinuousEvaluationCalculator(pp)
                    pgt = pgt.s([(pp.sample_dict[sample], sample)]).p([score]).t([target])
                    Chapter(pgt.get_description(), ch32).add_calculator(pgt)


        ch33 = Chapter('Marginal contributions', ch30)
        score = pp.scores[0]
        mcc = MarginalContributionsCalculator(pp)
        mcc = mcc.s([(pp.sample_dict[sample], sample) for sample in pp.sample_ordering]).t(pp.targets).p(cont_pred + pp.predictors_woe).sc([score])
        Chapter(mcc.get_description(), ch33).add_calculator(mcc)

        ch34 = Chapter('Transition matrix', ch30)
        for sample in pp.sample_ordering:
            tc = TransitionCalculator(pp)
            tc = tc.s([(pp.sample_dict[sample], sample)]).sc(pp.scores)
            Chapter(tc.get_description(), ch34).add_calculator(tc)

    def print_title(self, title):
        from datetime import datetime
        display(HTML("<a name='TOC'><h1>" + title + "</h1></a>"))
        display(HTML("<p>"+str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + "</p>"))
        display(HTML('<br>'))

    def print_summary(self):        
        def get_summary(ch, lvl=1):    
            if lvl <=4:
                display(HTML('<a href=' + '\'#' + ch.get_hash() + '\'' + ' style="margin-left: ' + str(40*lvl) +'px">' + ch.name + '</a>'))
            if len(ch.get_subchapters()) == 0:
                None
            else:
                for i in ch.get_subchapters():
                    get_summary(i, lvl+1)
                    
        get_summary(self.root)
        display(HTML('<br>'))
        
    def print_documentation(self): 
        def dcp(ch, lvl=1):    
                display(HTML('<a name='  + ch.get_hash() +' href=#TOC'+'><h'+str(lvl)+'>' + ch.name +'</h'+str(lvl)+ '></a>'))
                if len(ch.get_subchapters()) == 0:
                    display(ch.get_calculators()[0].calculate().get_visualization().get_table())
                    display(HTML('<br>'))
                else:
                    for i in ch.get_subchapters():
                        dcp(i, lvl+1)
                        
        dcp(self.root)