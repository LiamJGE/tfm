from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
from rdflib import Graph, Namespace, URIRef, Literal
import numpy.random as random
from rdflib.namespace import OWL, RDF, RDFS, XSD
import os
import json

class GraphBuilder():
    def __init__(self, config):
        self.graph  = Graph() 
        self.DBP = Namespace("http://dbpedia.org/resource/")
        self.DBPO = Namespace("http://dbpedia.org/ontology/")
        self.config = config
        self.uri_df = pd.read_csv(config['URI_DF'], sep='\t')

    def run(self):
        self.parse_schema()
        self.populate_graph()
        
        if self.config['GRAPH_MOD'] == 'class':
            self.label_as_class_of_entities()
        elif self.config['GRAPH_MOD'] == 'class_corr' or self.config['GRAPH_MOD'] == 'class_corr_less' or self.config['GRAPH_MOD'] == 'class_corr_less_less' or self.config['GRAPH_MOD'] == 'real_class_corr':
            self.label_as_class_of_entities()
        elif self.config['GRAPH_MOD'] == 'property':
            self.label_as_property_between_entities()
        elif self.config['GRAPH_MOD'] == 'property_corr' or self.config['GRAPH_MOD'] == 'property_corr_less'  or self.config['GRAPH_MOD'] == 'property_corr_less_less' or self.config['GRAPH_MOD'] == 'real_property_corr':
            self.corr_feat_as_property_between_entities()
        elif self.config['GRAPH_MOD'] == 'attribute':
            self.label_as_attribute_of_entities()
        elif self.config['GRAPH_MOD'] == 'attribute_corr':
            self.corr_feat_as_attribute_of_entities()

        self.save_graph_with_config(directory='../data/'+self.config['DATASET']+'/'+self.config['QUERY_TYPE']+'_query_'+self.config['GRAPH_MOD']+'mod')
        # self.save_graph_with_config(directory='../data/'+self.config['DATASET']+'/'+self.config['QUERY_TYPE']+'_query_'+self.config['GRAPH_MOD']+'mod_less_props')


    def construct_query_full(self, uris_of_interest):
        '''
        Returns properties and classes for URIs of interest. Filters out image related and wikipedia 
        link properties due to irrelevance.

        Parameters:
        - uris_of_interest: List of URIs to query
        '''
        uris_str = " ".join(f"<{uri}>" for uri in uris_of_interest)

        # if self.config['DATASET'] == 'forbes':
            # query = f"""
            # CONSTRUCT {{
            #     ?s ?p ?o .
            # }}
            # WHERE
            #     {{
            #         VALUES ?s {{ {uris_str} }}
            #         ?s ?p ?o .
            #         FILTER (REGEX(STR(?p), "income|equity|numberOfEmployees|assets|revenue|abstract", "i") || ?p = rdf:type )
            #     }}
            # """
        # elif self.config['DATASET'] == 'aaup':
        #     query = f"""
        #     CONSTRUCT {{
        #         ?s ?p ?o .
        #     }}
        #     WHERE
        #         {{
        #             VALUES ?s {{ {uris_str} }}
        #             ?s ?p ?o .
        #             FILTER (REGEX(STR(?p), "numberOfStudents|state|academicAffiliations|athleticsAffiliations|endowment|numberOfPostgraduateStudents|numberOfUndergraduateStudents|budget|abstract", "i") || ?p = rdf:type )
        #         }}
        #     """
        # else:
    
        query = f"""
            CONSTRUCT {{
                ?s ?p ?o .
            }}
            WHERE
                {{
                    VALUES ?s {{ {uris_str} }}
                    ?s ?p ?o .
                    FILTER (!regex(str(?p), "wiki", "i") && !regex(str(?p), "pic", "i") && !regex(str(?p), "thumb", "i") && !regex(str(?p), "alt", "i"))
                }}
        """

            #     UNION
            #     {{
            #         VALUES ?s {{ {uris_str} }}
            #         ?s ?p ?o .
            #         FILTER (isLiteral(?o) && langMatches(lang(?o), "en"))
            #         FILTER (!regex(str(?p), "wiki", "i") && !regex(str(?p), "pic", "i") && !regex(str(?p), "thumb", "i") && !regex(str(?p), "alt", "i"))
            #     }}
            # }}

        return query
    
    def construct_query_simple(self, uris_of_interest):
        '''
        Returns classes of DBpedia ontology and properties of type dbo:product and dbo:type for URIs of interest.
        It can be considered to return a simplified or filtered version of DBpedia KG.

        Parameters:
        - uris_of_interest: List of URIs to query
        '''

        uris_str = " ".join(f"<{uri}>" for uri in uris_of_interest)
    
        query = f"""
                    CONSTRUCT {{
                        ?s ?p ?o .
                    }}
                    WHERE {{
                        VALUES ?s {{ {uris_str} }}
                        ?s ?p ?o .
                        FILTER(?p = rdf:type)
                    }}
                """
                        # FILTER(STRSTARTS(STR(?p), STR(dbo:product)) || STRSTARTS(STR(?p), STR(dbo:type)) || (?p = rdf:type && STRSTARTS(STR(?o), STR("http://dbpedia.org/ontology"))))

        return query
    
    def execute_query(self, query):
        '''
            Executes query at DBpedia SPARQL endpoint.

            Parameters:
            - query: Query to execute
        '''
        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        sparql.addParameter("default-graph-uri", "http://dbpedia.org")
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)

        results = sparql.query().convert()
        return results
    
    def parse_schema(self):
        '''
        Adds schema to graph.

        Parameters:
        - schema: File where the schema is found.
        '''
        self.graph.parse(self.config['SCHEMA'], format='xml')

    def get_uris_of_interest(self):
        '''
        Return list of URIs of interest.
        '''
        return list(set(self.uri_df['uri'].values))


    def populate_graph(self):
        # Bind namespaces
        self.graph.bind("dbp", self.DBP)
        self.graph.bind("dbpo", self.DBPO)

        visited_uris = set()
        batch_size = 40
        uris_of_interest = self.get_uris_of_interest()

        for j in range(1):
            if j != 0:
                uris_of_interest = new_uris_of_interest
            new_uris_of_interest = []
            visited_uris.update(uris_of_interest)
            print(f'On iteration {j} the number of URIs to visit is: {len(uris_of_interest)}')
            
            for i in range(0, len(uris_of_interest), batch_size):
                # Construct the SPARQL query
                if self.config['QUERY_TYPE'] == 'full':
                    sparql_query = self.construct_query_full(uris_of_interest[i:i+batch_size]) if i+batch_size < len(uris_of_interest) else self.construct_query_full(uris_of_interest[i:len(uris_of_interest)])
                
                elif self.config['QUERY_TYPE'] == 'simple':
                    sparql_query = self.construct_query_simple(uris_of_interest[i:i+batch_size]) if i+batch_size < len(uris_of_interest) else self.construct_query_simple(uris_of_interest[i:len(uris_of_interest)])
                
                # Execute the query
                query_results = self.execute_query(sparql_query)

                # Add triples to the graph
                for triple in query_results['results']['bindings']:
                    subject = URIRef(triple['s']['value'])
                    predicate = URIRef(triple['p']['value'])
                    object_value = triple['o']['value']

                    # Check if object is a URI or a literal
                    if type(object_value) == str:
                        if object_value.startswith("http://dbpedia.org") or object_value.startswith("https://dbpedia.org"):
                            object_ = URIRef(object_value)
                            
                            # TO VISIT NEIGHBOURS OF URIS OF INTEREST
                            # random_add = (len(new_uris_of_interest) < 100)
                            # if random_add and object_value not in visited_uris and object_value not in all_popular_uris:
                            #     new_uris_of_interest.append(object_value)
                        else:
                            object_ = Literal(object_value)

                    else:
                        object_ = Literal(object_value)

                    if (subject, predicate, object_) not in self.graph:
                        self.graph.add((subject, predicate, object_))

                if i % 300 == 0:
                    print(i)


    def add_links(self, src, dst, label):
        '''
        Adds label property between two entities of the same class to the KG.

        Parameters:
        - src: Source entity node. AKA subject
        - dst: Destination entity node. AKA object
        - label: Label that both entities have.
        '''
        s = URIRef(src)
        o = URIRef(dst)
        p = URIRef("http://example.org/property/default/"+label)

        if (s,p,o) not in self.graph:
            self.graph.add((s,p,o))

    def find_links(self, x, col):
        '''
        For each entity passed as x, selects all the entities with the same label.

        Parameters:
        - x: Row of the uri_df
        '''
        sample_df = self.uri_df[self.uri_df[col] == x[col]]
        sample_df.apply(lambda y: self.add_links(x['uri'], y['uri'], x[col]), axis=1)

    def label_as_property_between_entities(self):
        '''
        Adds the label (target) as a property between all entities that belong to said label/class.
        '''
        self.uri_df.apply(lambda x: self.find_links(x, col='label'), axis=1)
    
    def corr_feat_as_property_between_entities(self):
        '''
        Adds the correlated feature as a property between all entities that belong to said label/class.
        '''
        self.uri_df.apply(lambda x: self.find_links(x, col='corr_feat'), axis=1)

    def add_label_classes(self, x, label):
        '''
        Adds the label (target) as a class. Each entity belonging to that label will be assigned to its
        appropriate class in the graph.
        '''
        s = x['uri'] if type(x['uri']) != str else URIRef(x['uri'])
        if (s, RDF.type, URIRef(str(self.DBPO)+label+'Class')) not in self.graph:
            self.graph.add((s, RDF.type, URIRef(str(self.DBPO)+label+'Class')))
    
    
    def label_as_class_of_entities(self):
        '''
        Adds the label (target) as a class. Each entity belonging to that label will be assigned to its
        appropriate class in the graph.
        '''
        if self.config['GRAPH_MOD'] == 'class':
            labels = list(set(self.uri_df['label'].values))
        else:
            labels = list(set(self.uri_df['corr_feat'].dropna().values))
        for label in labels:
            self.graph.add((URIRef(str(self.DBPO)+label+'Class'), RDF.type, RDFS.Class))
            self.graph.add((URIRef(str(self.DBPO)+label+'Class'), RDFS.subClassOf, self.DBPO.Organisation))
            if self.config['GRAPH_MOD'] == 'class':
                self.uri_df[self.uri_df['label'] == label].apply(lambda x: self.add_label_classes(x, label), axis=1)
            else:
                self.uri_df[self.uri_df['corr_feat'] == label].apply(lambda x: self.add_label_classes(x, label), axis=1)


    def add_label_property(self, x, label):
        '''
        Adds label as a proprty to the given uri of interest
        '''
        s = x['uri'] if type(x['uri']) != str else URIRef(x['uri'])
        if (s, self.DBPO.labelClass, Literal(label)) not in self.graph:
            self.graph.add((s, self.DBPO.labelClass, Literal(label)))


    def add_corr_feat_property(self, x, label):
        '''
        Adds correlated feature as a property to the given uri of interest
        '''
        s = x['uri'] if type(x['uri']) != str else URIRef(x['uri'])
        if (s, self.DBPO.corrFeat, Literal(label)) not in self.graph:
            self.graph.add((s, self.DBPO.corrFeat, Literal(label)))


    def label_as_attribute_of_entities(self):
        '''
        Adds label as a property of each entity of interest.
        '''
        self.graph.add((self.DBPO.labelClass, RDF.type, RDF.Property))
        self.graph.add((self.DBPO.labelClass, RDFS.domain, self.DBPO.Organisation))
        self.graph.add((self.DBPO.labelClass, RDFS.range, XSD.string))

        for label in list(set(self.uri_df['label'].values)):
                self.uri_df[self.uri_df['label'] == label].apply(lambda x: self.add_label_property(x, label), axis=1)

    def corr_feat_as_attribute_of_entities(self):
            '''
            Adds a correlated attribute as a property of each entity of interest.
            '''
            self.graph.add((self.DBPO.corrFeat, RDF.type, RDF.Property))
            self.graph.add((self.DBPO.corrFeat, RDFS.domain, self.DBPO.Organisation))
            self.graph.add((self.DBPO.corrFeat, RDFS.range, XSD.string))

            for label in list(set(self.uri_df['corr_feat'].values)):
                    self.uri_df[self.uri_df['corr_feat'] == label].apply(lambda x: self.add_corr_feat_property(x, label), axis=1)


    def serialise_graph(self, directory, modified=False):
        '''
        Serialises graph and exports it to given file location.
        '''
        # Serialize the graph as an OWL ontology
        owl_output = self.graph.serialize(format='xml')

        # Write the OWL ontology to a file
        with open(directory+'/graph.owl', 'w') as f:
            f.write(owl_output)
    

    def save_graph_with_config(self, directory):
        '''
        Saves configuration file with serialised graph.
        '''
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.serialise_graph(directory)

        with open(directory+'/config.json', 'w') as f:
            json.dump(self.config, f)