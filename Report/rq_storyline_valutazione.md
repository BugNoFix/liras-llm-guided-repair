# Storyline per collegare le research questions

## Idea narrativa centrale

La valutazione puo' essere raccontata come una progressione a livelli. Il punto di partenza non e' semplicemente misurare quale LLM produce piu' modelli corretti, ma capire fino a che punto una pipeline automatica possa trasformare una specifica naturale in un artefatto formale effettivamente verificabile.

La storia sperimentale puo' quindi seguire quattro domande consecutive:

1. gli LLM riescono a produrre un modello LIRAs valido gia' dalla specifica naturale?
2. quando falliscono, il feedback del compilatore e di UPPAAL aiuta davvero a recuperare il risultato?
3. questa capacita' resta stabile quando cambiano gli scenari?
4. quanto costa ottenere questi risultati, e quali errori rimangono fuori dalla portata del repair?

In questo modo, Effectiveness, Robustness, Efficiency e Failure Modes non sono sezioni indipendenti, ma parti dello stesso ragionamento: prima si misura la capacita' grezza dei modelli, poi il contributo del repair, poi la stabilita' del comportamento, poi il costo e i limiti strutturali.

## Research questions proposte

### RQ1 - Effectiveness

**To what extent can LLMs generate valid and verifiable LIRAs models from natural-language specifications?**

Questa e' la domanda principale della tesi. Stabilisce il criterio di successo end-to-end: non basta generare testo plausibile, e non basta produrre un DSL sintatticamente vicino alla grammatica. Il risultato deve attraversare l'intera pipeline: generazione del modello LIRAs, compilazione, eventuale esportazione/trasformazione e verifica tramite UPPAAL.

La sezione puo' iniziare mostrando il tasso di successo complessivo per modello. Qui conviene distinguere esplicitamente tre livelli:

- **syntactic validity**, quando il modello supera il compilatore LIRAs;
- **verifiability**, quando il modello puo' essere esportato e analizzato dalla toolchain;
- **end-to-end success**, quando l'intera pipeline produce un risultato utilizzabile.

Questa distinzione e' importante perche' anticipa uno dei risultati chiave: alcuni errori sono puramente sintattici e quindi recuperabili con feedback locale, mentre altri sono semantici o modellistici e richiedono una validazione piu' forte.

### RQ1.1 - Repair effectiveness

**How effective is an iterative compiler/UPPAAL-guided repair loop in recovering failed generations?**

Dopo aver misurato la generazione iniziale, la domanda naturale e': gli errori iniziali sono davvero definitivi? Questa sezione introduce il repair loop come secondo livello della pipeline.

La narrazione puo' confrontare:

- successi ottenuti al primo tentativo;
- successi ottenuti dopo uno o piu' cicli di repair;
- fallimenti rimasti dopo il limite massimo di iterazioni.

Qui va aggiunta in modo esplicito anche la parte di **syntactic repair**. Il compilatore fornisce feedback preciso su errori locali, quindi e' particolarmente adatto a correggere violazioni grammaticali o uso errato di costrutti del DSL. UPPAAL, invece, intercetta problemi piu' vicini alla semantica del modello, come time lock o comportamenti non verificabili.

La tesi puo' quindi sostenere una tesi intermedia: il repair guidato da tool non rende automaticamente affidabili tutti i modelli, ma sposta diversi fallimenti dalla categoria "generazione non valida" alla categoria "artefatto recuperato con supervisione automatica".

### RQ2 - Robustness

**How does model performance vary across different specification scenarios?**

Una volta stabilito se la pipeline funziona, bisogna capire se funziona in modo stabile. Questa sezione sposta l'attenzione dal modello allo scenario.

Il passaggio narrativo e': un buon tasso medio non basta, perche' una pipeline per specifiche formali deve comportarsi in modo prevedibile anche quando cambiano struttura, vincoli temporali, numero di entita' o combinazioni di azioni.

I grafici per scenario possono essere letti come una misura della fragilita' del sistema:

- scenari semplici mostrano la capacita' base del modello di seguire la grammatica;
- scenari con piu' vincoli temporali o interazioni rivelano errori semantici;
- scenari che richiedono azioni specifiche o costrutti come `follow` mettono in evidenza limiti nel grounding tra linguaggio naturale e DSL.

Questa RQ e' il ponte tra risultati quantitativi e failure analysis: se alcuni scenari falliscono in modo sistematico, allora il problema non e' solo il modello, ma l'interazione tra descrizione naturale, vincoli del DSL e feedback disponibile.

### RQ3 - Efficiency

**How many iterations, feedback cycles and tokens are required to obtain a valid solution?**

Questa parte risponde a una domanda pratica: anche quando la pipeline funziona, quanto costa usarla?

Conviene dividere Efficiency in tre sottodomande.

#### RQ3.1 - Iterations

**How many generation/repair iterations and feedback cycles are needed to reach a valid solution?**

Questa metrica misura la distanza tra generazione iniziale e artefatto valido. Un modello che ha molti successi ma richiede sempre diversi cicli di repair e' meno pratico di un modello che produce risultati validi rapidamente.

#### RQ3.2 - Token cost

**What is the average token cost per model, considering both successful and failed runs?**

Qui e' importante includere anche i fallimenti, perche' nella pratica consumano comunque risorse. Il costo medio non deve essere calcolato solo sui casi riusciti, altrimenti si sottostima il costo reale della pipeline.

#### RQ3.3 - Scenario efficiency

**How does token efficiency vary across specification scenarios?**

La metrica finale puo' essere definita come:

```text
efficiency = successful_scenarios / total_tokens
```

e normalizzata come:

```text
runs_successful_per_100k_tokens = efficiency * 100000
```

Questa normalizzazione rende confrontabili modelli e scenari diversi. Il messaggio da far passare e' che l'efficienza non coincide con il solo costo: un modello economico in token ma incapace di completare scenari complessi puo' avere efficienza bassa, mentre un modello piu' costoso puo' essere competitivo se produce piu' successi end-to-end.

### RQ4 - Failure modes

**What are the most common causes of failure in the end-to-end LLM-guided LIRAs generation pipeline?**

Questa sezione chiude la storia spiegando cosa i numeri non dicono da soli. Dopo effectiveness, robustness ed efficiency, l'analisi qualitativa dei fallimenti serve a capire perche' la pipeline si rompe.

Le categorie osservate possono essere organizzate cosi':

- **invalid action grounding**: uso di `stop` come action, nonostante non sia un'azione valida nel DSL;
- **syntactic misuse of DSL constructs**: uso non corretto di `follow`;
- **semantic/model-checking failures**: time lock o errori rilevati nella fase UPPAAL;
- **repair stagnation**: il modello riceve feedback ma non modifica sostanzialmente il DSL.

Esempi da discutere:

- **Gemma**: in 10 run i DSL generati contenevano `stop` come action; durante i cicli di repair il testo rimaneva sostanzialmente invariato. Questo e' un caso di fallimento sia nel grounding delle azioni sia nella capacita' di usare il feedback.
- **GPT-OSS**: 1 run con uso sintatticamente scorretto di `follow`, indicando un errore localizzato nel costrutto DSL.
- **Qwen 9B**: 5 run con errore su `follow` e 2 run con combinazione di errore su `follow` ed errore semantico. Questo suggerisce che il problema non e' solo grammaticale: in alcuni casi la correzione sintattica non basta.
- **Qwen 36B**: 3 run fallite per errore semantico dopo il limite di 5 cicli e 7 run fallite al primo ciclo per errore su `follow`. Questo separa chiaramente fallimenti recuperabili sintatticamente da fallimenti che resistono al repair.
- **GLM 5.2**: 4 run con errore su `follow`, confermando che questo costrutto e' una fonte ricorrente di fragilita'.

La conclusione della sezione dovrebbe essere che i fallimenti non sono casuali. Si concentrano su pochi punti critici: mapping delle azioni, costrutti DSL specifici e proprieta' temporali. Questo rende il risultato utile anche progettualmente, perche' indica dove intervenire: prompt piu' vincolanti sulle azioni ammesse, repair specializzato per `follow`, controlli sintattici preventivi e feedback UPPAAL piu' strutturato per i time lock.

## Ordine consigliato nel capitolo Evaluation

1. **Experimental setup**: modelli, scenari, numero di run, limite di iterazioni, metriche.
2. **RQ1 - End-to-end effectiveness**: successo complessivo e distinzione tra validita' sintattica e verificabilita'.
3. **RQ1.1 - Repair effectiveness**: quanti fallimenti iniziali vengono recuperati, con focus su syntactic repair.
4. **RQ2 - Robustness across scenarios**: variazione delle prestazioni per scenario.
5. **RQ3 - Efficiency**: iterazioni, feedback cycles, token medi e successi per 100k token.
6. **RQ4 - Failure modes**: classificazione qualitativa degli errori.
7. **Discussion**: cosa emerge dai risultati e quali implicazioni ha per pipeline LLM + formal verification.

## Paragrafo ponte pronto da usare

The evaluation is structured as a progressive analysis of the proposed LLM-guided LIRAs generation pipeline. First, we assess its effectiveness, measuring whether LLMs can produce models that are not only syntactically valid, but also usable by the downstream verification toolchain. We then isolate the contribution of the iterative repair loop, studying how often compiler and UPPAAL feedback can recover initially invalid generations. Since aggregate success rates may hide scenario-specific weaknesses, we next analyze robustness across different natural-language specifications. Finally, we evaluate the practical cost of the approach in terms of iterations and token usage, and we complement the quantitative results with a qualitative analysis of recurring failure modes. This organization reflects the end-to-end nature of the problem: a generated LIRAs model is useful only if it can be produced reliably, repaired when necessary, and verified within a reasonable computational budget.

## Paragrafo conclusivo pronto da usare

Overall, the results suggest that LLM-based generation of LIRAs models is feasible, but only when treated as a tool-guided process rather than as a one-shot translation task. Compiler feedback is effective for several syntactic errors and can recover a subset of initially failed generations, while UPPAAL-based validation exposes issues that are not visible at the DSL syntax level. At the same time, performance varies across scenarios and models, and the token cost of failed repair attempts must be considered when evaluating practical usability. The failure analysis shows that remaining errors are concentrated around a small number of recurring causes, especially invalid action grounding, incorrect use of the `follow` construct, and unresolved time-lock conditions. These findings motivate more specialized repair strategies and stronger constraints in future prompt and validation design.
