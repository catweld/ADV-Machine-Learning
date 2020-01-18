#!/usr/bin/env python
# coding: utf-8

# In[13]:


import argparse
import numpy as np
import matplotlib.pyplot as plt
from Tree import TreeMixture, Tree
import math
import sys
from Kruskal_v1 import Graph
import networkx as nx
import dendropy
import pickle

def save_results(loglikelihood, topology_array, theta_array, filename):
    """ This function saves the log-likelihood vs iteration values,
        the final tree structure and theta array to corresponding numpy arrays. """

    likelihood_filename = filename + "_em_loglikelihood.npy"
    topology_array_filename = filename + "_em_topology.npy"
    theta_array_filename = filename + "_em_theta.npy"
    print("Saving log-likelihood to ", likelihood_filename, ", topology_array to: ", topology_array_filename,
          ", theta_array to: ", theta_array_filename, "...")
    np.save(likelihood_filename, loglikelihood)
    np.save(topology_array_filename, topology_array)
    np.save(theta_array_filename, theta_array)

def responsibilities(num_clusters,samples,theta_list,topology_list,pi):
    num =  np.zeros(num_clusters*samples.shape[0]).reshape(samples.shape[0],num_clusters)
    denom = np.zeros(samples.shape[0])
    resp = np.zeros(num_clusters*samples.shape[0]).reshape(samples.shape[0],num_clusters)
    for n in range(samples.shape[0]):
        for k in range(num_clusters):
            p_init = theta_list[k][0][samples[n][0]]
            for s in range(1,samples.shape[1]):
                if(s==1):
                    p = p_init*theta_list[k][s][samples[n][int(topology_list[k][s])]][samples[n][s]]
                else:
                    p = p*theta_list[k][s][samples[n][int(topology_list[k][s])]][samples[n][s]]
            num[n,k] = pi[k]*p
            denom[n] += num[n,k]
    for n in range(samples.shape[0]):
        for k in range(num_clusters):
            resp[n,k] = (num[n,k])/(denom[n]+epsilon)
    return resp

def q_parts1(num_clusters,samples,resp):    
    N_ind1 = np.zeros((num_clusters,samples.shape[1],samples.shape[1],2,2))
    q_denom1 = np.zeros((num_clusters,samples.shape[1],samples.shape[1]))
    for k in range(num_clusters):
        for t in range(samples.shape[1]):
            for s in range(samples.shape[1]):
                for a in range(0,2):
                    for b in range(0,2):
                        for n in range(samples.shape[0]):
                            if ((samples[n,s] == a) & (samples[n,t] == b)):
                                I = 1
                            else:
                                I = 0
                            N_ind1[k,t,s,a,b] += resp[n,k]*I
                q_denom1[k,t,s] = np.sum(N_ind1[k,t,s,:,:])
    return N_ind1, q_denom1

def q_parts0(num_clusters,samples,resp):
    N_ind0 = np.zeros((num_clusters,samples.shape[1],2))
    q_denom0 = np.zeros((num_clusters,samples.shape[1]))
    for k in range(num_clusters):
        for s in range(samples.shape[1]):
            for a in range(0,2):
                for n in range(samples.shape[0]):
                    if ((samples[n,s] == a)):
                        I = 1
                    else:
                        I = 0
                    N_ind0[k,s,a] += resp[n,k]*I
                q_denom0[k,s] += N_ind0[k,s,a]
    return N_ind0, q_denom0
    
def q_parts1_cond(num_clusters,samples,resp):    
    N_ind1 = np.zeros((num_clusters,samples.shape[1],samples.shape[1],2,2))
    q_denom1 = np.zeros((num_clusters,samples.shape[1],samples.shape[1],2))
    for k in range(num_clusters):
        for t in range(samples.shape[1]):
            for s in range(samples.shape[1]):
                for a in range(0,2):
                    for b in range(0,2):
                        for n in range(samples.shape[0]):
                            if ((samples[n,s] == a) & (samples[n,t] == b)):
                                I = 1
                            else:
                                I = 0
                            N_ind1[k,t,s,a,b] += resp[n,k]*I
                        q_denom1[k,t,s,a] += N_ind1[k,t,s,a,b]
    return N_ind1, q_denom1

def q0(k,s,a,N_ind0,q_denom0):
    q_ind = N_ind0[k,s,a]/q_denom0[k,s]
    return q_ind

def q1(k,t,s,a,b,N_ind1,q_denom1):
    q_ind = N_ind1[k,t,s,a,b]/q_denom1[k,t,s]
    return q_ind

def I_Info(k,t,s,N_ind1,N_ind0,q_denom0,q_denom1,num_clusters,samples):
    Info = 0
    for a in range(0,2):
        for b in range(0,2):
            q1_joint = q1(k,t,s,a,b,N_ind1,q_denom1)
            q0_a = q0(k,s,a,N_ind0,q_denom0)
            q0_b = q0(k,t,b,N_ind0,q_denom0)
            if q1_joint != 0:
                Info = Info + q1_joint*np.log((q1_joint)/(q0_a*q0_b))
    return Info

def log_likelihood(num_clusters,samples,theta_list,topology_list,pi):
    likelihood = np.zeros(num_clusters*samples.shape[0]).reshape(samples.shape[0],num_clusters)
    loglikelihood = 0
    for n in range(samples.shape[0]):
        for k in range(num_clusters):
            for s in range(1,samples.shape[1]):
                p_init = theta_list[k][0][samples[n][0]]
                if(s==1):
                    p = p_init*theta_list[k][s][samples[n][int(topology_list[k][s])]][samples[n][s]]
                else:
                    p = p*theta_list[k][s][samples[n][int(topology_list[k][s])]][samples[n][s]]
                likelihood[n,k] = pi[k]*(p+epsilon)
    loglikelihood = np.log(np.sum(likelihood))
    return loglikelihood

def q_parts1cond(t,s,a,b,samples,resp):
    denom = 0
    num = 0
    for n in range(samples.shape[0]):
        if ((samples[n,s] == a) & (samples[n,t] == b)):
            I = 1
        else:
            I = 0
        num += resp[n]*I
    
    for n in range(samples.shape[0]):
        if (samples[n,s] == a):
            I = 1
        else:
            I = 0
        denom += resp[n]*I
    result = (num+epsilon)/(denom+epsilon)
    return result

def em_algorithm(seed_val, samples, num_clusters, max_num_iter=30):
    best_likelihood = -100000000.0
    best_seed = 0
    for seedno in range(100):
        np.random.seed(seedno)
        #print("Running EM algorithm...")
        tm = TreeMixture(num_clusters=num_clusters, num_nodes=samples.shape[1])
        tm.simulate_pi(seedno)
        tm.simulate_trees(seedno)
        tm.sample_mixtures(num_samples=samples.shape[0], seed_val = seedno)

        topology_list = []
        theta_list = []
        for k in range(num_clusters):
            topology_list.append(tm.clusters[k].get_topology_array())
            theta_list.append(tm.clusters[k].get_theta_array())

        topology_list = np.array(topology_list)
        theta_list = np.array(theta_list)
        loglikelihood = np.zeros(max_num_iter)
        pi = tm.pi

        for it in range(max_num_iter):
            #print("start iteration",it)
        #1: compute responsibilities
            resp = responsibilities(num_clusters,samples,theta_list,topology_list,pi)
            #print(resp)
        #2: set pi' = sum(r[n,k]/N)
            pi = np.zeros(num_clusters)
            pi_newdenom = np.sum(resp)
            for k in range(num_clusters):
                pi[k] = np.sum(resp[:,k])/pi_newdenom
            #print(pi)

        #3: calculate mutual information between x[s] and x[t]
            N_ind1,q_denom1 = q_parts1(num_clusters,samples,resp)
            N_ind0,q_denom0 = q_parts0(num_clusters,samples,resp)

        #4: set Tau'[k] as maximum spanning tree in G[k]
        ##PACKAGE NETWORKX USED TO CONVERT MAXIMUM SPANNING TREE TO TOPOLOGY
            trees = [Graph(samples.shape[1]),Graph(samples.shape[1]),Graph(samples.shape[1]),Graph(samples.shape[1])]
            weights = np.zeros((num_clusters,samples.shape[1],samples.shape[1]))
            MST = [Graph(samples.shape[1]),Graph(samples.shape[1]),Graph(samples.shape[1]),Graph(samples.shape[1])]
            for k in range(num_clusters):
                for s in range(samples.shape[1]):
                    for t in range(samples.shape[1]):
                        weights[k,s,t] = I_Info(k,s,t,N_ind1,N_ind0,q_denom0,q_denom1,num_clusters,samples)
                        #print(weights)
                        trees[k].addEdge(s,t,weights[k,s,t])
                MST[k] = trees[k].maximum_spanning_tree()
            tree_graphs = [nx.Graph(),nx.Graph(),nx.Graph(),nx.Graph()]
            treearray = [nx.Graph(),nx.Graph(),nx.Graph(),nx.Graph()]
            for k in range(num_clusters):
                for u_of_edge,v_of_edge,weight in MST[k]:
                    tree_graphs[k].add_edge(u_of_edge=u_of_edge,v_of_edge=v_of_edge)
                treearray[k] = list(nx.bfs_edges(G=tree_graphs[k],source=0))

            tau_new = topology_list
            for k in range(num_clusters):
                for s in range(0,len(treearray[k])):
                    parent = treearray[k][s][0]
                    child = treearray[k][s][1]
                    tau_new[k][child] = parent

        #5: set Theta'[k](X[r])
            theta_new = theta_list

            for k in range(num_clusters):
                theta_new[k][0][:] = [q0(k,0,0,N_ind0,q_denom0),q0(k,0,1,N_ind0,q_denom0)]
                for s in range(1,samples.shape[1]):
                    for a in range(0,2):
                        for b in range(0,2):
                            theta_new[k][s][a][b] = q_parts1cond(s,int(tau_new[k][s]),a,b,samples,resp[:,k])
                            
        #6: calculate log-likelihood
            theta_list = theta_new
            topology_list = tau_new
            loglikelihood[it] = log_likelihood(num_clusters,samples,theta_list,topology_list,pi)
            #print("best_likelihood = ",best_likelihood, "loglikelihood = ",loglikelihood[9])
        if best_likelihood < loglikelihood[25]:
            print(best_likelihood, ">", loglikelihood[25])
            best_likelihood = loglikelihood[25]
            best_seed = seedno
            
    print("seed val = ",best_seed)
    "repeat algorithm after finding best seed"
    
    np.random.seed(best_seed)
    #print("Running EM algorithm...")
    # TODO: Implement EM algorithm here.
    tm = TreeMixture(num_clusters=num_clusters, num_nodes=samples.shape[1])
    tm.simulate_pi(best_seed)
    tm.simulate_trees(best_seed)
    tm.sample_mixtures(num_samples=samples.shape[0], seed_val = best_seed)
    
    topology_list = []
    theta_list = []
    for k in range(num_clusters):
        topology_list.append(tm.clusters[k].get_topology_array())
        theta_list.append(tm.clusters[k].get_theta_array())

    #loglikelihood = np.array(loglikelihood)
    topology_list = np.array(topology_list)
    theta_list = np.array(theta_list)
    #start iterations
    loglikelihood = np.zeros(max_num_iter)
    pi = tm.pi
    
    for it in range(max_num_iter):
            #print("start iteration",it)
        #1: compute responsibilities
            resp = responsibilities(num_clusters,samples,theta_list,topology_list,pi)
            #print(resp)
        #2: set pi' = sum(r[n,k]/N)
            pi = np.zeros(num_clusters)
            pi_newdenom = np.sum(resp)
            for k in range(num_clusters):
                pi[k] = np.sum(resp[:,k])/pi_newdenom
            #print(pi)

        #3: calculate mutual information between x[s] and x[t]
            N_ind1,q_denom1 = q_parts1(num_clusters,samples,resp)
            N_ind0,q_denom0 = q_parts0(num_clusters,samples,resp)

        #4: set Tau'[k] as maximum spanning tree in G[k]
            trees = [Graph(samples.shape[1]),Graph(samples.shape[1]),Graph(samples.shape[1]),Graph(samples.shape[1])]
            weights = np.zeros((num_clusters,samples.shape[1],samples.shape[1]))
            MST = [Graph(samples.shape[1]),Graph(samples.shape[1]),Graph(samples.shape[1]),Graph(samples.shape[1])]
            for k in range(num_clusters):
                for s in range(samples.shape[1]):
                    for t in range(samples.shape[1]):
                        weights[k,s,t] = I_Info(k,s,t,N_ind1,N_ind0,q_denom0,q_denom1,num_clusters,samples)
                        trees[k].addEdge(s,t,weights[k,s,t])
                MST[k] = trees[k].maximum_spanning_tree()

            ##PACKAGE NETWORKX USED TO CONVERT MAXIMUM SPANNING TREE TO TOPOLOGY
            tree_graphs = [nx.Graph(),nx.Graph(),nx.Graph(),nx.Graph()]
            treearray = [nx.Graph(),nx.Graph(),nx.Graph(),nx.Graph()]
            for k in range(num_clusters):
                for u_of_edge,v_of_edge,weight in MST[k]:
                    tree_graphs[k].add_edge(u_of_edge=u_of_edge,v_of_edge=v_of_edge)
                treearray[k] = list(nx.bfs_edges(G=tree_graphs[k],source=0))

            tau_new = topology_list
            for k in range(num_clusters):
                for s in range(0,len(treearray[k])):
                    parent = treearray[k][s][0]
                    child = treearray[k][s][1]
                    tau_new[k][child] = parent

        #5: set Theta'[k](X[r])
            theta_new = theta_list

            for k in range(num_clusters):
                theta_new[k][0][:] = [q0(k,0,0,N_ind0,q_denom0),q0(k,0,1,N_ind0,q_denom0)]
                for s in range(1,samples.shape[1]):
                    for a in range(0,2):
                        for b in range(0,2):
                            theta_new[k][s][a][b] = q_parts1cond(s,int(tau_new[k][s]),a,b,samples,resp[:,k])
                            
        #6: calculate log-likelihood
            theta_list = theta_new
            topology_list = tau_new
            loglikelihood[it] = log_likelihood(num_clusters,samples,theta_list,topology_list,pi)
    
    print("topology_list = ",topology_list)
    print(loglikelihood)
    return loglikelihood, np.array(topology_list), theta_list

def main():
    print("\n1. Load samples from txt file.\n")
    default_sample_filename = 'q_2_5_tm_10node_20sample_4clusters.pkl_samples.txt'
    default_output_filename = 'q_2_5_tm_10node_20sample_4clusters_results'
    default_num_clusters = 4
    default_seed_val = 42
    default_real_values_filename = "q_2_5_tm_10node_20sample_4clusters.pkl_tree_0_topology.npy"
    
    
    samples = np.loadtxt(default_sample_filename, delimiter="\t", dtype=np.int32)
    num_samples, num_nodes = samples.shape
    print("\tnum_samples: ", num_samples, "\tnum_nodes: ", num_nodes)
    print("\tSamples: \n", samples)

    print("\n2. Run EM Algorithm.\n")

    loglikelihood, topology_array, theta_array = em_algorithm(default_seed_val, samples, num_clusters=default_num_clusters)
    
    
    print("\n3. Save, print and plot the results.\n")
    print("top array 0 = ",topology_array[0,:])
    save_results(loglikelihood, topology_array, theta_array, default_output_filename)
    save_results(loglikelihood, topology_array[0,:], theta_array[0,:], "q_2_5_tm_10node_20sample_4clusters0_results")
    save_results(loglikelihood, topology_array[1,:], theta_array[1,:], "q_2_5_tm_10node_20sample_4clusters1_results")
    save_results(loglikelihood, topology_array[2,:], theta_array[2,:], "q_2_5_tm_10node_20sample_4clusters2_results")
    save_results(loglikelihood, topology_array[3,:], theta_array[3,:], "q_2_5_tm_10node_20sample_4clusters3_results")

    for i in range(default_num_clusters):
        print("\n\tCluster: ", i)
        print("\tTopology: \t", topology_array[i])
        print("\tTheta: \t", theta_array[i])
    
    
    Theta_Expected = np.array(theta_array.shape)
    Topology_Expected = np.zeros((default_num_clusters,samples.shape[1]))
    
    filename = "q_2_5_tm_10node_20sample_4clusters.pkl_tree_0_theta.npy"
    theta_expected0 = np.load(filename)
    filename = "q_2_5_tm_10node_20sample_4clusters.pkl_tree_0_topology.npy"
    topology_expected0 = np.load(filename)
    filename = "q_2_5_tm_10node_20sample_4clusters.pkl_pi.npy"
    pi_expected = np.load(filename)
    
    filename = "q_2_5_tm_10node_20sample_4clusters.pkl_tree_1_theta.npy"
    theta_expected1 = np.load(filename)
    filename = "q_2_5_tm_10node_20sample_4clusters.pkl_tree_1_topology.npy"
    topology_expected1 = np.load(filename)
    
    filename = "q_2_5_tm_10node_20sample_4clusters.pkl_tree_2_theta.npy"
    theta_expected2 = np.load(filename)
    filename = "q_2_5_tm_10node_20sample_4clusters.pkl_tree_2_topology.npy"
    topology_expected2 = np.load(filename)
    
    filename = "q_2_5_tm_10node_20sample_4clusters.pkl_tree_3_theta.npy"
    theta_expected3 = np.load(filename)
    filename = "q_2_5_tm_10node_20sample_4clusters.pkl_tree_3_topology.npy"
    topology_expected3 = np.load(filename)
    Theta_Expected = theta_array
    
    thetafiles = [theta_expected0,theta_expected1,theta_expected2,theta_expected3]
    topologyfiles = [topology_expected0,topology_expected1,topology_expected2,topology_expected3]
    for k in range(default_num_clusters):
        Theta_Expected[k][:][:][:] = thetafiles[k]
        Topology_Expected[k,:] = topologyfiles[k]
    loglikelihood_expected = np.zeros(len(loglikelihood))
    for iterations in range(len(loglikelihood)):
        loglikelihood_expected[iterations] = log_likelihood(default_num_clusters,samples,Theta_Expected,Topology_Expected,pi_expected)
    
    print("loglikelihood_expected = ",loglikelihood_expected[0])
    plt.figure(figsize=(8, 3))
    plt.subplot(121)
    plt.plot(np.exp(loglikelihood_expected), label='Expected')
    plt.plot(np.exp(loglikelihood), label='Estimated')
    plt.ylabel("Likelihood of Mixture")
    plt.xlabel("Iterations")
    plt.subplot(122)
    plt.plot(loglikelihood_expected, label='Expected')
    plt.plot(loglikelihood, label='Estimated')
    plt.ylabel("Log-Likelihood of Mixture")
    plt.xlabel("Iterations")
    plt.legend(loc=(1.04, 0))
    plt.show()

    print("\n4. Retrieve real results and compare.\n")
    if default_real_values_filename != "":
        print("\tComparing the results with real values...")

        print("\t4.1. Make the Robinson-Foulds distance analysis.\n")
        print("Hello World!")
        print("This file demonstrates the usage of the DendroPy module and its functions.")

        print("\n1. Tree Generations\n")
        print("\n1.1. Create two random birth-death trees and print them:\n")

        tns = dendropy.TaxonNamespace()
        num_leaves = 5

        t1 = dendropy.simulate.treesim.birth_death_tree(birth_rate=1.0, death_rate=0.5, num_extant_tips=num_leaves,
                                                        taxon_namespace=tns)
        t2 = dendropy.simulate.treesim.birth_death_tree(birth_rate=1.0, death_rate=0.2, num_extant_tips=num_leaves,
                                                        taxon_namespace=tns)
        print("\tTree 1: ", t1.as_string("newick"))
        t1.print_plot()
        print("\tTree 2: ", t2.as_string("newick"))
        t2.print_plot()

        print("\n2. Compare Trees\n")
        print("\n2.1. Compare tree with itself and print Robinson-Foulds (RF) distance:\n")

        print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t1, t1))

        print("\n2.2. Compare different trees and print Robinson-Foulds (RF) distance:\n")

        print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t1, t2))

        print("\n3. Load Trees from Newick Files and Compare:\n")
        print("\n3.1 Load trees from Newick files:\n")

        # If you want to compare two trees, make sure you specify the same Taxon Namespace!
        tns = dendropy.TaxonNamespace()

        filename = "q_2_5_tm_10node_20sample_4clusters.pkl_tree_0_newick.txt"
        with open(filename, 'r') as input_file:
            newick_str = input_file.read()
        t0 = dendropy.Tree.get(data=newick_str, schema="newick", taxon_namespace=tns)
        print("\tTree 0: ", t0.as_string("newick"))
        t0.print_plot()

        filename = "q_2_5_tm_10node_20sample_4clusters.pkl_tree_1_newick.txt"
        with open(filename, 'r') as input_file:
            newick_str = input_file.read()
        t1 = dendropy.Tree.get(data=newick_str, schema="newick", taxon_namespace=tns)
        print("\tTree 1: ", t1.as_string("newick"))
        t1.print_plot()

        filename = "q_2_5_tm_10node_20sample_4clusters.pkl_tree_2_newick.txt"
        with open(filename, 'r') as input_file:
            newick_str = input_file.read()
        t2 = dendropy.Tree.get(data=newick_str, schema="newick", taxon_namespace=tns)
        print("\tTree 2: ", t2.as_string("newick"))
        t2.print_plot()
        
        filename = "q_2_5_tm_10node_20sample_4clusters.pkl_tree_3_newick.txt"
        with open(filename, 'r') as input_file:
            newick_str = input_file.read()
        t3 = dendropy.Tree.get(data=newick_str, schema="newick", taxon_namespace=tns)
        print("\tTree 3: ", t3.as_string("newick"))
        t3.print_plot()

        print("\n3.2 Compare trees and print Robinson-Foulds (RF) distance:\n")
        
        print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t0, t1))
        print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t0, t2))
        print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t1, t2))
        
        filename = "q_2_5_tm_10node_20sample_4clusters0_results_em_topology.npy"
        topology_list = np.load(filename)  
        rt0 = Tree()
        rt0.load_tree_from_direct_arrays(np.array(topology_list))
        rt0 = dendropy.Tree.get(data=rt0.newick, schema="newick", taxon_namespace=tns)
        print("\tInferred Tree 0: ", rt0.as_string("newick"))
        rt0.print_plot()

        filename = "q_2_5_tm_10node_20sample_4clusters1_results_em_topology.npy"
        topology_list = np.load(filename)  
        rt1 = Tree()
        rt1.load_tree_from_direct_arrays(np.array(topology_list))
        rt1 = dendropy.Tree.get(data=rt1.newick, schema="newick", taxon_namespace=tns)
        print("\tInferred Tree 1: ", rt1.as_string("newick"))
        rt1.print_plot()
        
        filename = "q_2_5_tm_10node_20sample_4clusters2_results_em_topology.npy"
        topology_list = np.load(filename)  
        rt2 = Tree()
        rt2.load_tree_from_direct_arrays(np.array(topology_list))
        rt2 = dendropy.Tree.get(data=rt2.newick, schema="newick", taxon_namespace=tns)
        print("\tInferred Tree 2: ", rt2.as_string("newick"))
        rt2.print_plot()
        
        filename = "q_2_5_tm_10node_20sample_4clusters3_results_em_topology.npy"
        topology_list = np.load(filename)  
        rt3 = Tree()
        rt3.load_tree_from_direct_arrays(np.array(topology_list))
        rt3 = dendropy.Tree.get(data=rt3.newick, schema="newick", taxon_namespace=tns)
        print("\tInferred Tree 3: ", rt3.as_string("newick"))
        rt3.print_plot()

        print("\n4.2 Compare trees and print Robinson-Foulds (RF) distance:\n")

        print("\tt0 vs inferred trees")
        print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t0, rt0))
        print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t0, rt1))
        print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t0, rt2))

        print("\tt1 vs inferred trees")
        print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t1, rt0))
        print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t1, rt1))
        print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t1, rt2))

        print("\tt2 vs inferred trees")
        print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t2, rt0))
        print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t2, rt1))
        print("\tRF distance: \t", dendropy.calculate.treecompare.symmetric_difference(t2, rt2))
        
        

        print("\t4.2. Make the likelihood comparison.\n")


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




