#define FORCE_IMPORT_ARRAY
#include "growcut_fast.hpp"

template <class T>
using ndarray = xt::pyarray<T>;

void print2d(vector<vector<int> > &vec){
	for (int i = 0; i < vec.size(); i++)
	{
		for (int j = 0; j < vec[i].size(); j++)
		{
			cout<<vec[i][j]<<'\t';
		}
		cout<<endl;
	}
}

vector<vector<int> > get_offsets(const int dim){
	const int N = pow(3.,dim);
	vector< vector<int> > offsets(N,vector<int>(dim,0));
	for (int i = 1; i < N; i++)
	{
		bool flag = true;
		for (int d = dim-1; d >= 0; d--) {
			if (flag) {
				int temp = (offsets[i-1][d]+1) + 1;
				flag = temp >= 3;
				offsets[i][d] = temp % 3 - 1;
			}
			else {
				offsets[i][d] = offsets[i-1][d];
			}
		}
	}
	return offsets;
}

vector<vector<int> > get_p_neighbours(vector<int> p, vector<int> shape, bool exclude_p = true){
	const int dim = p.size();
	vector<vector<int> > offsets = get_offsets(dim);
	vector<vector<int> > neighbours;
	if (exclude_p) offsets.erase(offsets.begin());
	
	for (int i = 0; i < offsets.size(); i++)
	{
		vector<int> temp(dim, 0);
		bool flag = true;
		for (int j = 0; j < dim; j++) {
			temp[j] = offsets[i][j] + p[j];
			if (temp[j] < 0 || temp[j] >= shape[j]) {
				flag = false;
			}
		}
		if (flag) {
			neighbours.push_back(temp);
		}
	}
	return neighbours;
}

ndarray<double> get_img_neighbours(ndarray<double> const& mask){
    const int dim = mask.dimension();
    auto shape = mask.shape();

    // padding mask
    auto mask_pad = ndarray<double>::from_shape(shape);
    copy(mask.cbegin(), mask.cend(), mask_pad.begin());
    vector<int> shape_add;
    int pad_width = 1;
    for (int i = 0; i < dim; i++) shape_add.push_back(shape[i]);
    
    for (int i = 0; i < dim; i++){
        shape_add[i] = pad_width;
        auto add = 0 * xt::ones<double>(shape_add);
        mask_pad = xt::concatenate(xt::xtuple(add, mask_pad, add), i);
        shape_add[i] = shape[i] + pad_width * 2;
    }

    auto idx = xt::from_indices(xt::nonzero(mask_pad));

    vector<vector<int> > offsets = get_offsets(dim);
    offsets.erase(offsets.begin());

    for (int i = 0; i < offsets.size(); i++){
        for (int j = 0; j < idx.shape()[1]; j++){
            auto view_idx = xt::view(idx, xt::all(), j);
            auto shift_idx = view_idx + xt::adapt(offsets[i], {dim});
            auto temp = xt::adapt(offsets[i], {dim});
            xt::xstrided_slice_vector sv;
            for (int d = 0; d < dim; d++) sv.push_back(shift_idx(d));
            auto view = xt::strided_view(mask_pad, sv);
            view = 1;
        }
    }
 
    xt::xstrided_slice_vector sv_mask;
    for (int i = 0; i < dim; i++) sv_mask.push_back(xt::range(1, shape[i]+1));
    auto view_mask = xt::strided_view(mask_pad, sv_mask);

    return view_mask;
}

ndarray<double> growcut_cpu(ndarray<double> const& img, ndarray<double> const& seeds, ndarray<double> &labPre, ndarray<double> &distPre, bool newSeg = true, bool verbose = true){
    auto dim = img.dimension();
    auto shape = img.shape();

    auto labCrt = ndarray<double>::from_shape(shape);
    auto distCrt = ndarray<double>::from_shape(shape);
    auto mask = ndarray<bool>::from_shape(shape);

    if (newSeg){
        // for neSweg, use copy seed label as current label
        copy(seeds.cbegin(), seeds.cend(), labCrt.begin());
        // distCrt=0 at seeds, distCrt=np.inf elsewhere
        distCrt.fill(numeric_limits<double>::infinity());
        filtration(distCrt, seeds > 0) = 0.;
    }
    else
    {
        labCrt.fill(0.);
        distCrt.fill(numeric_limits<double>::infinity());
        // if not newSeg, only use the seeds with different labels from labPre
        mask = seeds > 0 && xt::not_equal(seeds, labPre);
        // update labCrt with seed label
        auto view_l = xt::filter(labCrt, mask);
        auto view_s = xt::filter(seeds, mask);
        copy(view_s.cbegin(), view_s.cend(), view_l.begin());
        filtration(distCrt, mask) = 0;
    }

    // initialzation of fibonacci heap
    FibonacciHeap<double, tuple<vector<int>, xt::xstrided_slice_vector> > fh;
    // choose non-labeled pixels and their neighbors as heap nodes
    mask = xt::equal(labCrt, 0);
    mask = get_img_neighbours(mask);

    auto idx = xt::from_indices(xt::nonzero(mask));

    map<vector<int>, node<double, tuple<vector<int>, xt::xstrided_slice_vector> >*> heapNodes;

    for (int i = 0; i < idx.shape()[1]; i++){
        xt::xstrided_slice_vector psv;
        vector<int> pidx;
        for (int d = 0; d < dim; d++){
            psv.push_back(idx(d, i));
            pidx.push_back(idx(d, i));
        }
        auto view_d = xt::strided_view(distCrt, psv);
        double key;
        copy(view_d.cbegin(), view_d.cbegin()+1, &key);
        heapNodes.insert(make_pair(pidx, fh.insert(key, make_tuple(pidx, psv))));
    }

    int count = 0;

    // segmentation/refinement loop
    while (!fh.isEmpty()){
        auto pnode = fh.getMinimumNode();
        auto pinfo = pnode -> getValue();
        auto pidx = get<0>(pinfo);
        auto psv = get<1>(pinfo);
        if (verbose) cout<<"extracting node"<<endl;
        fh.removeMinimum();
        if (verbose) cout<<"updating node"<<endl;
        auto view_dc = xt::strided_view(distCrt, psv);
        double distCrt_p;
        copy(view_dc.cbegin(), view_dc.cbegin()+1, &distCrt_p);

        auto view_dp = xt::strided_view(distPre, psv);
        double distPre_p;
        copy(view_dp.cbegin(), view_dp.cbegin()+1, &distPre_p);
        
        auto view_lc = xt::strided_view(labCrt, psv);
        double labCrt_p;
        copy(view_lc.cbegin(), view_lc.cbegin()+1, &labCrt_p);

        auto view_lp = xt::strided_view(labPre, psv);
        double labPre_p;
        copy(view_lp.cbegin(), view_lp.cbegin()+1, &labPre_p);
        
        auto view_img = xt::strided_view(img, psv);
        double img_p;
        copy(view_img.cbegin(), view_img.cbegin()+1, &img_p);

        if (!newSeg)
        {
            if (isinf(distCrt_p)) break;
            if (distCrt_p > distPre_p)
            {
                view_dc = distPre_p;
                view_lc = labPre_p;
                continue;
            }
        }

        vector<int> vector_shape;
        for (int d = 0; d < dim; d++) vector_shape.push_back(shape[d]);
        auto neighbours = get_p_neighbours(pidx, vector_shape, true);
        for (int i = 0; i < neighbours.size(); i++)
        {
            xt::xstrided_slice_vector psv_n;
            for (int d = 0; d < dim; d++) psv_n.push_back(neighbours[i][d]);

            auto view_dc_n = xt::strided_view(distCrt, psv_n);
            double distCrt_pn;
            copy(view_dc_n.cbegin(), view_dc_n.cbegin()+1, &distCrt_pn);

            auto view_lc_n = xt::strided_view(labCrt, psv_n);
            double labCrt_pn;
            copy(view_lc_n.cbegin(), view_lc_n.cbegin()+1, &labCrt_pn);

            auto view_img_n = xt::strided_view(img, psv_n);
            double img_pn;
            copy(view_img_n.cbegin(), view_img_n.cbegin()+1, &img_pn);

            auto node_n = heapNodes[pidx];
            double dist = distCrt_p + sqrt(pow(img_pn - img_p,2));
            if (dist < distCrt_pn)
            {
                view_dc_n = dist;
                view_lc_n = labCrt_p;
                // update fiponacci heap
                fh.decreaseKey(node_n, dist);
            }
        }

        if (verbose)
        {
            // regular dijkstra
            cout<<"-----------Dijkastra-----------"<<endl;
            cout<<"Current point: "<< count <<"\t Distance from seed: "<<distCrt_p<<"\t Seed Label: "<<labCrt_p<<endl;
            count++;
        }
    }
    cout << "segmentation finished!" << endl;

    if (!newSeg)
    {
        // get updated points
        mask = labCrt < numeric_limits<double>::infinity();
        // update local states
        auto view_dc = xt::filter(distCrt, mask);
        auto view_dp = xt::filter(distPre, mask);
        auto view_lc = xt::filter(labCrt, mask);
        auto view_lp = xt::filter(labPre, mask);

        copy(view_dc.cbegin(), view_dc.cend(), view_dp.begin());
        copy(view_lc.cbegin(), view_lc.cend(), view_lp.begin());
        copy(distPre.cbegin(), distPre.cend(), distCrt.begin());
        copy(labPre.cbegin(), labPre.cend(), labCrt.begin());        
    }
    copy(distCrt.cbegin(), distCrt.cend(), distPre.begin());
    copy(labCrt.cbegin(), labCrt.cend(), labPre.begin()); 

    return labPre;
}

PYBIND11_MODULE(growcut_fast, m){
    xt::import_numpy();
    m.doc() = "fast growcut implementation";
    m.def("growcut_cpu", &growcut_cpu, "cpu verison of growcut algorithms with shortest path");
}