# Proposta di struttura per la tesi

Data raccolta: 2026-07-12

Contesto del progetto: pipeline per generare modelli LIRAs da scenari in linguaggio naturale usando LLM, ripararli con feedback del compilatore, esportarli in XML, adattare query e validare con UPPAAL/verifyta.

## Campione Politesi consultato

Ho preso un campione di tesi magistrali Politesi selezionate dal corso "Computer Science and Engineering - Ingegneria Informatica" e da ricerche interne su LLM, generative AI, testing, model checking e formal verification. Ho usato gli indici/sommari dei PDF aperti per capire la struttura, non il contenuto specifico.

| # | Tesi | Link | Pattern strutturale utile |
|---|---|---|---|
| 1 | State of art and best practices in generative conversational AI | https://www.politesi.polimi.it/handle/10589/141797 | Introduzione -> background ampio -> case study/data preprocessing -> model learning/evaluation -> risultati qualitativi e quantitativi |
| 2 | DILLEMA: metamorphic testing for deep learning using diffusion and large language models | https://www.politesi.polimi.it/handle/10589/214006 | Introduzione -> related work -> background -> solution -> implementation -> evaluation -> conclusion |
| 3 | Effectiveness and optimization of large language models in natural language querying for MongoDB data retrieval | https://www.politesi.polimi.it/handle/10589/223237 | Problem statement -> research questions -> literature review -> experiments, con metodi, infrastruttura, risultati e appendici sui prompt |
| 4 | Detecting fact-conflicting hallucinations through the use of large language models | https://www.politesi.polimi.it/handle/10589/225573 | Background/motivazioni -> related works -> research questions -> approcci ed esperimenti -> conclusioni che rispondono alle RQ |
| 5 | Large language models for fact-checking over tables | https://www.politesi.polimi.it/handle/10589/226776 | Introduction -> background -> related works -> research questions -> data -> approach -> experiments -> conclusion |
| 6 | AI in education: leveraging generative models for exercise generation and resolution | https://www.politesi.polimi.it/handle/10589/235498 | Problem description -> goal/contribution -> background LLM -> implemented solution -> metrics -> conclusion/future directions |
| 7 | Formal verification of parametric timed automata with TABEC | https://www.politesi.polimi.it/handle/10589/218025 | Introduction -> state of the art -> theoretical background -> tool/algorithm -> random testing -> experimental results |
| 8 | Verification and synthesis of Infrastructure-as-Code through satisfiability modulo theories | https://www.politesi.polimi.it/handle/10589/210113 | Background -> model checker/tool internals -> requirements DSL -> synthesis -> performance/example -> conclusions |
| 9 | Counterexample extraction in a context-free model checker | https://www.politesi.polimi.it/handle/10589/208561 | Theory chapters -> practical implementation -> counterexample completion -> experimental results |
| 10 | POMC. Toward a model checking tool for operator precedence languages | https://www.politesi.polimi.it/handle/10589/164972 | Preliminary concepts -> formal verification -> tool architecture -> experiments -> future work |
| 11 | Design, implementation, and pilot testing of an automated method to characterize mobile health apps' topical areas by extracting information from the web | https://www.politesi.polimi.it/handle/10589/141793 | Introduction/background -> materials and methods -> results -> discussion/conclusions |
| 12 | TZ logger. A multi-platform TEE logger | https://www.politesi.polimi.it/handle/10589/141800 | Motivations/goals/challenges -> approach -> implementation -> exploration -> related work -> discussion |

## Pattern osservati

Le tesi piu' vicine al tuo tipo di lavoro seguono quasi sempre questa sequenza:

1. Motivazione e obiettivi: perche' il problema conta, quali limiti hanno gli approcci attuali, quali contributi porta la tesi.
2. Background: concetti necessari per leggere il resto, spesso separati dallo stato dell'arte.
3. Related work / stato dell'arte: non solo elenco di paper, ma confronto che porta al gap.
4. Research questions: nelle tesi sperimentali moderne e' molto utile averle esplicite.
5. Approccio o metodologia: definizione del problema, dati/scenari, metriche, disegno sperimentale.
6. Architettura e implementazione: pipeline, moduli, scelte tecniche, logging, riproducibilita'.
7. Valutazione: setup, metriche, risultati, confronto tra varianti, analisi degli errori.
8. Discussione: cosa funziona, cosa no, limiti, minacce alla validita'.
9. Conclusioni e sviluppi futuri.
10. Appendici: prompt completi, grammatica DSL, scenari, query, configurazioni, tabelle complete.

## Struttura consigliata

### Front matter

- Abstract in inglese
- Sommario in italiano
- Ringraziamenti, se vuoi
- Indice
- Lista figure e tabelle
- Glossario / acronimi: LIRAs, LLM, DSL, UPPAAL, PTA/TA, XML, verifyta, RQ

### 1. Introduction

1.1 Context and motivation  
1.2 Problem statement  
1.3 Goals and research questions  
1.4 Contributions  
1.5 Thesis structure  

Possibili research questions:

- RQ1: In che misura gli LLM generano modelli LIRAs sintatticamente validi da scenari in linguaggio naturale?
- RQ2: Quanto migliora il feedback del compilatore la validita' sintattica e strutturale dei modelli generati?
- RQ3: Quali errori rimangono dopo la compilazione e vengono intercettati dalla validazione UPPAAL?
- RQ4: Come cambiano successo, costo e robustezza al variare di modello, prompt, shot e strategia di repair?

### 2. Background

2.1 Domain-specific languages and model-driven engineering  
2.2 LIRAs: concetti, sintassi e ruolo nel progetto  
2.3 Timed automata, UPPAAL e verifyta  
2.4 Large language models per generazione di codice/specifiche  
2.5 Prompting, few-shot learning e feedback-based repair  

Qui devi spiegare solo quello che serve per leggere la pipeline. Lo stato dell'arte dettagliato va nel capitolo successivo.

### 3. State of the Art

3.1 Natural-language-to-DSL e model generation  
3.2 LLM per specifiche formali e logiche temporali  
3.3 LLM-guided repair, self-debugging e compiler feedback  
3.4 LLM e formal verification/model checking  
3.5 Gap analysis: perche' serve una pipeline generate-repair-verify per LIRAs  

Questo capitolo deve chiudersi dicendo chiaramente dove si inserisce il tuo contributo.

### 4. Problem Definition and Methodology

4.1 Input del sistema: scenari naturali, query e vincoli  
4.2 Output attesi: LIRAs valido, XML, query adattate, risultati UPPAAL  
4.3 Dataset sperimentale: scenari, query, baseline manuali, criteri di inclusione  
4.4 Variabili sperimentali: modelli, provider, prompt, shot, temperature, iterazioni  
4.5 Metriche: compile success, repair iterations, UPPAAL pass rate, probability delta, token/costo, tempo, error categories  
4.6 Protocollo sperimentale e riproducibilita'  

Questo e' il capitolo che evita l'effetto "ho costruito una pipeline e basta": rende il lavoro una tesi sperimentale.

### 5. LIRAs LLM-Guided Repair Pipeline

5.1 Architecture overview  
5.2 DSL generation from natural language  
5.3 Compiler-guided repair loop  
5.4 XML export  
5.5 Query adaptation  
5.6 UPPAAL/verifyta validation and feedback cycles  
5.7 Run metadata, logging and telemetry  
5.8 Dashboard and analysis utilities  
5.9 Implementation details and engineering choices  

Qui puoi usare direttamente la struttura reale del repository: `pipeline_runner.py`, `dsl_generator.py`, `query_adapter.py`, `SPs/`, `Scenarios/`, `Queries/`, `Runs/`, `Report/`.

### 6. Experimental Evaluation

6.1 Experimental setup  
6.2 Overview of executed runs  
6.3 Syntactic validity and compiler repair results  
6.4 XML export and query adaptation results  
6.5 UPPAAL validation results  
6.6 Comparison by model, prompt and shot configuration  
6.7 Cost, latency and token usage  
6.8 Qualitative error analysis  
6.9 Threats to validity  

Per il tuo progetto e' molto importante separare:

- errori sintattici/compilatore;
- errori di esportazione XML;
- errori nelle query;
- fallimenti UPPAAL;
- casi che compilano ma sono semanticamente sospetti.

### 7. Discussion

7.1 What compiler feedback fixes well  
7.2 What formal validation catches beyond compilation  
7.3 Syntax vs semantics in LLM-generated formal artifacts  
7.4 Practical recommendations for prompt and repair design  
7.5 Generalization to other DSLs and model-checking workflows  
7.6 Limitations  

Questo capitolo serve a far vedere maturita': non solo numeri, ma interpretazione.

### 8. Conclusions and Future Work

8.1 Summary of contributions  
8.2 Answers to the research questions  
8.3 Future work  

Future work possibili:

- benchmark piu' ampio di scenari;
- metriche semantiche piu' forti;
- feedback da counterexample UPPAAL piu' strutturato;
- prompt repair specializzati per categorie di errore;
- human-in-the-loop review;
- supporto ad altri DSL o altri model checker.

## Appendici consigliate

- Appendix A: grammatica LIRAs o subset usato nella tesi.
- Appendix B: prompt completi di generazione, repair e query adaptation.
- Appendix C: scenari e query usate negli esperimenti.
- Appendix D: configurazioni dei run.
- Appendix E: tabelle complete dei risultati.
- Appendix F: esempi di fallimento e riparazione.

## Versione compatta dell'indice

Se il relatore preferisce una tesi piu' snella:

1. Introduction
2. Background and Related Work
3. Methodology
4. Pipeline Design and Implementation
5. Experimental Evaluation
6. Discussion
7. Conclusions

La versione estesa e' migliore se vuoi dare peso sia allo stato dell'arte sia alla validazione formale.
