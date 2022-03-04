pragma Ada_2005;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings;
with System;
with stddef_h;
with library_types_h;

package nvgraph_h is

  -- * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
  -- *
  -- * NVIDIA CORPORATION and its licensors retain all intellectual property
  -- * and proprietary rights in and to this software, related documentation
  -- * and any modifications thereto.  Any use, reproduction, disclosure or
  -- * distribution of this software and related documentation without an express
  -- * license agreement from NVIDIA CORPORATION is strictly prohibited.
  -- *
  --  

  -- nvGRAPH status type returns  
   type nvgraphStatus_t is 
     (NVGRAPH_STATUS_SUCCESS,
      NVGRAPH_STATUS_NOT_INITIALIZED,
      NVGRAPH_STATUS_ALLOC_FAILED,
      NVGRAPH_STATUS_INVALID_VALUE,
      NVGRAPH_STATUS_ARCH_MISMATCH,
      NVGRAPH_STATUS_MAPPING_ERROR,
      NVGRAPH_STATUS_EXECUTION_FAILED,
      NVGRAPH_STATUS_INTERNAL_ERROR,
      NVGRAPH_STATUS_TYPE_NOT_SUPPORTED,
      NVGRAPH_STATUS_NOT_CONVERGED);
   pragma Convention (C, nvgraphStatus_t);  -- /usr/local/cuda-8.0/include/nvgraph.h:45

   function nvgraphStatusGetString (status : nvgraphStatus_t) return Interfaces.C.Strings.chars_ptr;  -- /usr/local/cuda-8.0/include/nvgraph.h:47
   pragma Import (C, nvgraphStatusGetString, "nvgraphStatusGetString");

  -- Opaque structure holding nvGRAPH library context  
   --  skipped empty struct nvgraphContext

   type nvgraphHandle_t is new System.Address;  -- /usr/local/cuda-8.0/include/nvgraph.h:51

  -- Opaque structure holding the graph descriptor  
   --  skipped empty struct nvgraphGraphDescr

   type nvgraphGraphDescr_t is new System.Address;  -- /usr/local/cuda-8.0/include/nvgraph.h:55

  -- Semi-ring types  
   type nvgraphSemiring_t is 
     (NVGRAPH_PLUS_TIMES_SR,
      NVGRAPH_MIN_PLUS_SR,
      NVGRAPH_MAX_MIN_SR,
      NVGRAPH_OR_AND_SR);
   pragma Convention (C, nvgraphSemiring_t);  -- /usr/local/cuda-8.0/include/nvgraph.h:64

  -- Topology types  
   type nvgraphTopologyType_t is 
     (NVGRAPH_CSR_32,
      NVGRAPH_CSC_32,
      NVGRAPH_COO_32);
   pragma Convention (C, nvgraphTopologyType_t);  -- /usr/local/cuda-8.0/include/nvgraph.h:72

  -- Default is unsorted.
  -- CSR
  -- CSC
   type nvgraphTag_t is 
     (NVGRAPH_DEFAULT,
      NVGRAPH_UNSORTED,
      NVGRAPH_SORTED_BY_SOURCE,
      NVGRAPH_SORTED_BY_DESTINATION);
   pragma Convention (C, nvgraphTag_t);  -- /usr/local/cuda-8.0/include/nvgraph.h:80

  -- n+1
   type nvgraphCSRTopology32I_st is record
      nvertices : aliased int;  -- /usr/local/cuda-8.0/include/nvgraph.h:83
      nedges : aliased int;  -- /usr/local/cuda-8.0/include/nvgraph.h:84
      source_offsets : access int;  -- /usr/local/cuda-8.0/include/nvgraph.h:85
      destination_indices : access int;  -- /usr/local/cuda-8.0/include/nvgraph.h:86
   end record;
   pragma Convention (C_Pass_By_Copy, nvgraphCSRTopology32I_st);  -- /usr/local/cuda-8.0/include/nvgraph.h:82

  -- nnz
  -- rowPtr
  -- colInd
   type nvgraphCSRTopology32I_t is access all nvgraphCSRTopology32I_st;  -- /usr/local/cuda-8.0/include/nvgraph.h:88

  -- n+1
   type nvgraphCSCTopology32I_st is record
      nvertices : aliased int;  -- /usr/local/cuda-8.0/include/nvgraph.h:91
      nedges : aliased int;  -- /usr/local/cuda-8.0/include/nvgraph.h:92
      destination_offsets : access int;  -- /usr/local/cuda-8.0/include/nvgraph.h:93
      source_indices : access int;  -- /usr/local/cuda-8.0/include/nvgraph.h:94
   end record;
   pragma Convention (C_Pass_By_Copy, nvgraphCSCTopology32I_st);  -- /usr/local/cuda-8.0/include/nvgraph.h:90

  -- nnz
  -- colPtr
  -- rowInd
   type nvgraphCSCTopology32I_t is access all nvgraphCSCTopology32I_st;  -- /usr/local/cuda-8.0/include/nvgraph.h:96

  -- n+1
   type nvgraphCOOTopology32I_st is record
      nvertices : aliased int;  -- /usr/local/cuda-8.0/include/nvgraph.h:99
      nedges : aliased int;  -- /usr/local/cuda-8.0/include/nvgraph.h:100
      source_indices : access int;  -- /usr/local/cuda-8.0/include/nvgraph.h:101
      destination_indices : access int;  -- /usr/local/cuda-8.0/include/nvgraph.h:102
      tag : aliased nvgraphTag_t;  -- /usr/local/cuda-8.0/include/nvgraph.h:103
   end record;
   pragma Convention (C_Pass_By_Copy, nvgraphCOOTopology32I_st);  -- /usr/local/cuda-8.0/include/nvgraph.h:98

  -- nnz
  -- rowInd
  -- colInd
   type nvgraphCOOTopology32I_t is access all nvgraphCOOTopology32I_st;  -- /usr/local/cuda-8.0/include/nvgraph.h:105

  -- Open the library and create the handle  
   function nvgraphCreate (handle : System.Address) return nvgraphStatus_t;  -- /usr/local/cuda-8.0/include/nvgraph.h:108
   pragma Import (C, nvgraphCreate, "nvgraphCreate");

  --  Close the library and destroy the handle   
   function nvgraphDestroy (handle : nvgraphHandle_t) return nvgraphStatus_t;  -- /usr/local/cuda-8.0/include/nvgraph.h:111
   pragma Import (C, nvgraphDestroy, "nvgraphDestroy");

  -- Create an empty graph descriptor  
   function nvgraphCreateGraphDescr (handle : nvgraphHandle_t; descrG : System.Address) return nvgraphStatus_t;  -- /usr/local/cuda-8.0/include/nvgraph.h:114
   pragma Import (C, nvgraphCreateGraphDescr, "nvgraphCreateGraphDescr");

  -- Destroy a graph descriptor  
   function nvgraphDestroyGraphDescr (handle : nvgraphHandle_t; descrG : nvgraphGraphDescr_t) return nvgraphStatus_t;  -- /usr/local/cuda-8.0/include/nvgraph.h:117
   pragma Import (C, nvgraphDestroyGraphDescr, "nvgraphDestroyGraphDescr");

  -- Set size, topology data in the graph descriptor   
   function nvgraphSetGraphStructure
     (handle : nvgraphHandle_t;
      descrG : nvgraphGraphDescr_t;
      topologyData : System.Address;
      TType : nvgraphTopologyType_t) return nvgraphStatus_t;  -- /usr/local/cuda-8.0/include/nvgraph.h:120
   pragma Import (C, nvgraphSetGraphStructure, "nvgraphSetGraphStructure");

  -- Query size and topology information from the graph descriptor  
   function nvgraphGetGraphStructure
     (handle : nvgraphHandle_t;
      descrG : nvgraphGraphDescr_t;
      topologyData : System.Address;
      TType : access nvgraphTopologyType_t) return nvgraphStatus_t;  -- /usr/local/cuda-8.0/include/nvgraph.h:123
   pragma Import (C, nvgraphGetGraphStructure, "nvgraphGetGraphStructure");

  -- Allocate numsets vectors of size V reprensenting Vertex Data and attached them the graph.
  -- * settypes[i] is the type of vector #i, currently all Vertex and Edge data should have the same type  

   function nvgraphAllocateVertexData
     (handle : nvgraphHandle_t;
      descrG : nvgraphGraphDescr_t;
      numsets : stddef_h.size_t;
      settypes : access library_types_h.cudaDataType_t) return nvgraphStatus_t;  -- /usr/local/cuda-8.0/include/nvgraph.h:127
   pragma Import (C, nvgraphAllocateVertexData, "nvgraphAllocateVertexData");

  -- Allocate numsets vectors of size E reprensenting Edge Data and attached them the graph.
  -- * settypes[i] is the type of vector #i, currently all Vertex and Edge data should have the same type  

   function nvgraphAllocateEdgeData
     (handle : nvgraphHandle_t;
      descrG : nvgraphGraphDescr_t;
      numsets : stddef_h.size_t;
      settypes : access library_types_h.cudaDataType_t) return nvgraphStatus_t;  -- /usr/local/cuda-8.0/include/nvgraph.h:131
   pragma Import (C, nvgraphAllocateEdgeData, "nvgraphAllocateEdgeData");

  -- Update the vertex set #setnum with the data in *vertexData, sets have 0-based index
  -- *  Conversions are not sopported so nvgraphTopologyType_t should match the graph structure  

   function nvgraphSetVertexData
     (handle : nvgraphHandle_t;
      descrG : nvgraphGraphDescr_t;
      vertexData : System.Address;
      setnum : stddef_h.size_t) return nvgraphStatus_t;  -- /usr/local/cuda-8.0/include/nvgraph.h:135
   pragma Import (C, nvgraphSetVertexData, "nvgraphSetVertexData");

  -- Copy the edge set #setnum in *edgeData, sets have 0-based index
  -- *  Conversions are not sopported so nvgraphTopologyType_t should match the graph structure  

   function nvgraphGetVertexData
     (handle : nvgraphHandle_t;
      descrG : nvgraphGraphDescr_t;
      vertexData : System.Address;
      setnum : stddef_h.size_t) return nvgraphStatus_t;  -- /usr/local/cuda-8.0/include/nvgraph.h:139
   pragma Import (C, nvgraphGetVertexData, "nvgraphGetVertexData");

  -- Convert the edge data to another topology
  --  

   function nvgraphConvertTopology
     (handle : nvgraphHandle_t;
      srcTType : nvgraphTopologyType_t;
      srcTopology : System.Address;
      srcEdgeData : System.Address;
      dataType : access library_types_h.cudaDataType_t;
      dstTType : nvgraphTopologyType_t;
      dstTopology : System.Address;
      dstEdgeData : System.Address) return nvgraphStatus_t;  -- /usr/local/cuda-8.0/include/nvgraph.h:143
   pragma Import (C, nvgraphConvertTopology, "nvgraphConvertTopology");

  -- Convert graph to another structure
  --  

   function nvgraphConvertGraph
     (handle : nvgraphHandle_t;
      srcDescrG : nvgraphGraphDescr_t;
      dstDescrG : nvgraphGraphDescr_t;
      dstTType : nvgraphTopologyType_t) return nvgraphStatus_t;  -- /usr/local/cuda-8.0/include/nvgraph.h:149
   pragma Import (C, nvgraphConvertGraph, "nvgraphConvertGraph");

  -- Update the edge set #setnum with the data in *edgeData, sets have 0-based index
  -- *  Conversions are not sopported so nvgraphTopologyType_t should match the graph structure  

   function nvgraphSetEdgeData
     (handle : nvgraphHandle_t;
      descrG : nvgraphGraphDescr_t;
      edgeData : System.Address;
      setnum : stddef_h.size_t) return nvgraphStatus_t;  -- /usr/local/cuda-8.0/include/nvgraph.h:153
   pragma Import (C, nvgraphSetEdgeData, "nvgraphSetEdgeData");

  -- Copy the edge set #setnum in *edgeData, sets have 0-based index
  -- * Conversions are not sopported so nvgraphTopologyType_t should match the graph structure  

   function nvgraphGetEdgeData
     (handle : nvgraphHandle_t;
      descrG : nvgraphGraphDescr_t;
      edgeData : System.Address;
      setnum : stddef_h.size_t) return nvgraphStatus_t;  -- /usr/local/cuda-8.0/include/nvgraph.h:157
   pragma Import (C, nvgraphGetEdgeData, "nvgraphGetEdgeData");

  -- create a new graph by extracting a subgraph given a list of vertices
  --  

   function nvgraphExtractSubgraphByVertex
     (handle : nvgraphHandle_t;
      descrG : nvgraphGraphDescr_t;
      subdescrG : nvgraphGraphDescr_t;
      subvertices : access int;
      numvertices : stddef_h.size_t) return nvgraphStatus_t;  -- /usr/local/cuda-8.0/include/nvgraph.h:161
   pragma Import (C, nvgraphExtractSubgraphByVertex, "nvgraphExtractSubgraphByVertex");

  -- create a new graph by extracting a subgraph given a list of edges
  --  

   function nvgraphExtractSubgraphByEdge
     (handle : nvgraphHandle_t;
      descrG : nvgraphGraphDescr_t;
      subdescrG : nvgraphGraphDescr_t;
      subedges : access int;
      numedges : stddef_h.size_t) return nvgraphStatus_t;  -- /usr/local/cuda-8.0/include/nvgraph.h:164
   pragma Import (C, nvgraphExtractSubgraphByEdge, "nvgraphExtractSubgraphByEdge");

  -- nvGRAPH Semi-ring sparse matrix vector multiplication
  --  

   function nvgraphSrSpmv
     (handle : nvgraphHandle_t;
      descrG : nvgraphGraphDescr_t;
      weight_index : stddef_h.size_t;
      alpha : System.Address;
      x_index : stddef_h.size_t;
      beta : System.Address;
      y_index : stddef_h.size_t;
      SR : nvgraphSemiring_t) return nvgraphStatus_t;  -- /usr/local/cuda-8.0/include/nvgraph.h:168
   pragma Import (C, nvgraphSrSpmv, "nvgraphSrSpmv");

  -- nvGRAPH Single Source Shortest Path (SSSP)
  -- * Calculate the shortest path distance from a single vertex in the graph to all other vertices.
  --  

   function nvgraphSssp
     (handle : nvgraphHandle_t;
      descrG : nvgraphGraphDescr_t;
      weight_index : stddef_h.size_t;
      source_vert : access int;
      sssp_index : stddef_h.size_t) return nvgraphStatus_t;  -- /usr/local/cuda-8.0/include/nvgraph.h:180
   pragma Import (C, nvgraphSssp, "nvgraphSssp");

  -- nvGRAPH WidestPath
  -- * Find widest path potential from source_index to every other vertices.
  --  

   function nvgraphWidestPath
     (handle : nvgraphHandle_t;
      descrG : nvgraphGraphDescr_t;
      weight_index : stddef_h.size_t;
      source_vert : access int;
      widest_path_index : stddef_h.size_t) return nvgraphStatus_t;  -- /usr/local/cuda-8.0/include/nvgraph.h:189
   pragma Import (C, nvgraphWidestPath, "nvgraphWidestPath");

  -- nvGRAPH PageRank
  -- * Find PageRank for each vertex of a graph with a given transition probabilities, a bookmark vector of dangling vertices, and the damping factor.
  --  

   function nvgraphPagerank
     (handle : nvgraphHandle_t;
      descrG : nvgraphGraphDescr_t;
      weight_index : stddef_h.size_t;
      alpha : System.Address;
      bookmark_index : stddef_h.size_t;
      has_guess : int;
      pagerank_index : stddef_h.size_t;
      tolerance : float;
      max_iter : int) return nvgraphStatus_t;  -- /usr/local/cuda-8.0/include/nvgraph.h:198
   pragma Import (C, nvgraphPagerank, "nvgraphPagerank");

  -- extern "C"  
end nvgraph_h;
