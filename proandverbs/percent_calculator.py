import pandas as pd
import spacy
import re

# cargamos el modelo con especificaciones: https://github.com/explosion/spacy-models/releases/tag/es_core_news_sm-3.7.0
nlp = spacy.load('es_core_news_sm')

# cargamos los datos
datos = pd.read_csv("rawdata.csv")

# limpiamos tantito los ensayos
datos['fu_ensayo_2'] = datos['fu_ensayo_2'].apply(str)

# aplicamos el modelo de spacy a cada uno de los ensayos
datos['doc'] = datos['fu_ensayo_2'].apply(nlp)

# NOTE: Como los modelos de morfología no dependen fuertemente de la ortografía, no es necesario corregir la ortografía de los ensayos


docs = datos['doc'].tolist()

# de la operación del modelo tomamos la información que es de interés, ej: 
# así se ve un token: ('convinados', 'ADJ', 'Gender=Masc|Number=Plur|VerbForm=Part')

datos['wordandtag'] = datos['doc'].apply(lambda x: [(token.text, token.tag_, str(token.morph)) for token in x])

wordandtag = datos['wordandtag'].tolist()

#DESCOMENTAR ESTO PARA QUE VISUALIZAR LAS CLASIFICACIONES DE LAS PALABRAS


# for wt in wordandtag:
#     for w, t, morph in wt:
#         if t in ['PRON', 'VERB', 'AUX', 'DET']:
#             print(w, t, morph)

# vamos a calcular el porcentaje del texto que está escrito en primera persona y en tercera persona

# vamos a usar reglas de real academia española para contabilizar el porcentaje
# https://www.rae.es/gtg/persona#
            

# de acuerdo con la RAE la persona se define como:
# Categoría gramatical que informa sobre los participantes del discurso y que se manifiesta 
# como propiedad flexiva de los pronombres personales, de los posesivos y del verbo

# por lo que solo vamos a centrarnos en los pronombres personales, los verbos y en los posesivos tanto adjetivos como pronombres
# los posesivos son un caso especial pues están en la categoría de DET

def calcularPersonaVerbo(token):
    """Esta función regresa un valor numerico dependiendo de la persona del verbo

    Args:
        token (str): verbo auxiliar o principal

    Returns:
        int: 1 si es primera persona, 3 si es tercera persona
    """
    # tercera persona plural
    if token[-1].lower() == 'n':
        return 3
    # primera persona plural
    if len(token) > 3 and token[-3:].lower() == 'mos' or token[-3:].lower() == 'nos':
        return 1
    
    # no es primera ni tercera persona plural
    return 0


def calcularPersonaPosesivo(token):
    """Esta función regresa un valor numerico dependiendo de la persona del posesivo

    Args:
        token (str): posesivo

    Returns:
        int: 1 si es primera persona, 3 si es tercera persona
    """
    
    primerapersona = ["mi", "mis", "mío", "mía", "míos", "mías", "nuestro", "nuestra", "nuestros", "nuestras", "mio", "mia", "mios", "mias"]
    tercerapersona = ["su", "sus", "suyo", "suya", "suyos", "suyas", "cuyo", "cuya", "cuyos", "cuyas"]

    if token.lower() in primerapersona:
        return 1
    
    if token.lower() in tercerapersona:
        return 3
    
    # caso especial donde la categoria DET no es posesivo
    return 0 

def calcularPersonaPronombre(token):
    """Esta función regresa un valor numerico dependiendo de la persona del pronombre

    Args:
        token (str): pronombre

    Returns:
        int: 1 si es primera persona, 3 si es tercera persona
    """
    primerapersona = ["yo", "me", "mí", "mi", "nosotros", "nosotras", "nos", "nuestro", "nuestra", "nuestros", "nuestras"]
    tercerapersona = ["él", "ella", "ello", "ellos", "ellas", "ustedes", "le", "lo", "la", "los", "las", "les", "se", "su", "sus", "suyo", "suya", "suyos", "suyas", "cuyo", "cuya", "cuyos", "cuyas", "el"]
    #tercerapersona += ["un", "una", "unos", "unas"] # podríamos agregar este caso especial para oraciones como "un mexicano" o "una mexicana"
    if token.lower() in primerapersona:
        return 1
    
    if token.lower() in tercerapersona:
        return 3
    
    return 0

# esta función es para verificar que la palabra sea válida, no en cuanto a ortografía sino que tenga caracteres válidos
def verifyword(token):
    spanish_char_regex = r"^[a-zA-ZáéíóúÁÉÍÓÚüÜñÑ]+$"
    return bool(re.match(spanish_char_regex, token))

#wordandtag = [eval(x) for x in wordandtag]
evaluations = {'PRON': calcularPersonaPronombre, 'DET': calcularPersonaPosesivo, 'VERB': calcularPersonaVerbo, 'AUX': calcularPersonaVerbo}
personclassified = []
primerapercentages = []
tercerapercentages = []

# iteramos sobre todos los ensayos
for wt in wordandtag:
    # variables para cada una de los ensayos
    temp = []
    allcounter = 0
    primeracounter = 0
    terceracounter = 0
    # iteramos sobre cada palabra, su clasificación y su morfología
    for w, t, morph in wt:
        # verificamos que sea una palabra válida y aporte información sobre en que persona está escrita
        if verifyword(w) and (t in ['PRON', 'VERB', 'AUX', 'DET'] or "person" in morph.lower()):
            personamanual = 0
            personaspacy = 0
            # si la palabra se puede clasificar con las reglas de la RAE (manualmente)
            if t in evaluations:
                personamanual = evaluations[t](w)

            # si la palabra se puede clasificar con spacy
            if "person" in morph.lower():
                # como morph contiene más información que solo la persona, vamos a buscar la persona
                pos = morph.find("Person")
                personaspacy = morph[pos + 7]
                personaspacy = int(personaspacy)

            # caso que hay conflictos entre las clasificaciones
            if personamanual != personaspacy and personamanual != 0 and personaspacy != 0:
                # le damos prioridad a la clasificación manual
                if personamanual == 1:
                    primeracounter += 1
                elif personamanual == 3: 
                    terceracounter += 1
            # no hay conflicto o alguno puso 0
            else:
                # no hay conflicto
                if personamanual == personaspacy:
                    if personamanual == 1:
                        primeracounter += 1
                    elif personamanual == 3: 
                        terceracounter += 1
                # si alguno puso 0 tomamos el valor del que no lo hizo
                else: 
                    if personamanual + personaspacy == 1:
                        primeracounter += 1
                    elif personamanual + personaspacy == 3:
                        terceracounter += 1
            # caso para ciertas palabras que todavía se pueden clasificar y no se han clasificado
            if personamanual == personaspacy == 0 and t == 'DET':
                personamanual = calcularPersonaPronombre(w)
            # guardamos estos valores para cada oración
            temp.append((w, t, personamanual, personaspacy))

            # solo contamos aquellas palabras que aportan información sobre la persona
            allcounter += 1

    
    # hay ensayos sin pronombre, verbo, auxiliar o posesivo, básicamente nan's
    if allcounter == 0:
        primerapercentages.append(0)
        tercerapercentages.append(0)
    else: 
        primerapercentages.append(round(primeracounter/allcounter, 2))
        tercerapercentages.append(round(terceracounter/allcounter, 2))

    personclassified.append(temp)

# guardamos los datos a un csv
datos['personclassified'] = personclassified
datos['firstpersonpercent'] = primerapercentages
datos['thirdpersonpercent'] = tercerapercentages

# quitamos las columnas que no nos interesan
del datos['doc']
del datos['fu_ensayo_2']
del datos['treat']
del datos['wordandtag']
# guardamos los datos
datos.to_csv("rawdatawithperson.csv", index=False)