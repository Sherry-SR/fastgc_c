// Author: Robin Message <Robin.Message@cl.cam.ac.uk>
// Modified by Rui Shen

// fibonacci.hpp

#ifndef FIBONACCI_H
#define FIBONACCI_H

template <class V, class T> class FibonacciHeap;

template <class V, class T> struct node {
private:
	node<V, T>* prev;
	node<V, T>* next;
	node<V, T>* child;
	node<V, T>* parent;
	V key;
	T value;
	int degree;
	bool marked;
public:
	friend class FibonacciHeap<V, T>;
	node<V, T>* getPrev() {return prev;}
	node<V, T>* getNext() {return next;}
	node<V, T>* getChild() {return child;}
	node<V, T>* getParent() {return parent;}
	V getKey() {return key;}
	T getValue() {return value;}
	bool isMarked() {return marked;}

	bool hasChildren() {return child;}
	bool hasParent() {return parent;}
};

template <class V, class T> class FibonacciHeap {
protected:
	node<V, T>* heap;
public:

	FibonacciHeap() {
		heap=_empty();
	}
	virtual ~FibonacciHeap() {
		if(heap) {
			_deleteAll(heap);
		}
	}
	node<V, T>* insert(V key, T value = NULL) {
		node<V, T>* ret=_singleton(key, value);
		heap=_merge(heap,ret);
		return ret;
	}
	void merge(FibonacciHeap& other) {
		heap=_merge(heap,other.heap);
		other.heap=_empty();
	}

	bool isEmpty() {
		return heap==NULL;
	}

	node<V, T>* getMinimumNode() {
		return heap;
	}

	V removeMinimum() {
		node<V, T>* old=heap;
		heap=_removeMinimum(heap);
		V ret=old->key;
		delete old;
		return ret;
	}

	void decreaseKey(node<V, T>* n,V key) {
		heap=_decreaseKey(heap,n,key);
	}

	node<V, T>* find(V key) {
		return _find(heap,key);
	}
private:
	node<V, T>* _empty() {
		return NULL;
	}

	node<V, T>* _singleton(V key, T value = NULL) {
		node<V, T>* n=new node<V, T>;
		n->key=key;
		n->value = value;
		n->prev=n->next=n;
		n->degree=0;
		n->marked=false;
		n->child=NULL;
		n->parent=NULL;
		return n;
	}

	node<V, T>* _merge(node<V, T>* a,node<V, T>* b) {
		if(a==NULL)return b;
		if(b==NULL)return a;
		if(a->key>b->key) {
			node<V, T>* temp=a;
			a=b;
			b=temp;
		}
		node<V, T>* an=a->next;
		node<V, T>* bp=b->prev;
		a->next=b;
		b->prev=a;
		an->prev=bp;
		bp->next=an;
		return a;
	}

	void _deleteAll(node<V, T>* n) {
		if(n!=NULL) {
			node<V, T>* c=n;
			do {
				node<V, T>* d=c;
				c=c->next;
				_deleteAll(d->child);
				delete d;
			} while(c!=n);
		}
	}
	
	void _addChild(node<V, T>* parent,node<V, T>* child) {
		child->prev=child->next=child;
		child->parent=parent;
		parent->degree++;
		parent->child=_merge(parent->child,child);
	}

	void _unMarkAndUnParentAll(node<V, T>* n) {
		if(n==NULL)return;
		node<V, T>* c=n;
		do {
			c->marked=false;
			c->parent=NULL;
			c=c->next;
		}while(c!=n);
	}

	node<V, T>* _removeMinimum(node<V, T>* n) {
		_unMarkAndUnParentAll(n->child);
		if(n->next==n) {
			n=n->child;
		} else {
			n->next->prev=n->prev;
			n->prev->next=n->next;
			n=_merge(n->next,n->child);
		}
		if(n==NULL)return n;
		node<V, T>* trees[64]={NULL};
		
		while(true) {
			if(trees[n->degree]!=NULL) {
				node<V, T>* t=trees[n->degree];
				if(t==n)break;
				trees[n->degree]=NULL;
				if(n->key<t->key) {
					t->prev->next=t->next;
					t->next->prev=t->prev;
					_addChild(n,t);
				} else {
					t->prev->next=t->next;
					t->next->prev=t->prev;
					if(n->next==n) {
						t->next=t->prev=t;
						_addChild(t,n);
						n=t;
					} else {
						n->prev->next=t;
						n->next->prev=t;
						t->next=n->next;
						t->prev=n->prev;
						_addChild(t,n);
						n=t;
					}
				}
				continue;
			} else {
				trees[n->degree]=n;
			}
			n=n->next;
		}
		node<V, T>* min=n;
		node<V, T>* start=n;
		do {
			if(n->key<min->key)min=n;
			n=n->next;
		} while(n!=start);
		return min;
	}

	node<V, T>* _cut(node<V, T>* heap,node<V, T>* n) {
		if(n->next==n) {
			n->parent->child=NULL;
		} else {
			n->next->prev=n->prev;
			n->prev->next=n->next;
			n->parent->child=n->next;
		}
		n->next=n->prev=n;
		n->marked=false;
		return _merge(heap,n);
	}

	node<V, T>* _decreaseKey(node<V, T>* heap,node<V, T>* n,V key) {
		if(n->key<key)return heap;
		n->key=key;
		if(n->parent) {
			if(n->key<n->parent->key) {
				heap=_cut(heap,n);
				node<V, T>* parent=n->parent;
				n->parent=NULL;
				while(parent!=NULL && parent->marked) {
					heap=_cut(heap,parent);
					n=parent;
					parent=n->parent;
					n->parent=NULL;
				}
				if(parent!=NULL && parent->parent!=NULL)parent->marked=true;
			}
		} else {
			if(n->key < heap->key) {
				heap = n;
			}
		}
		return heap;
	}

	node<V, T>* _find(node<V, T>* heap,V key) {
		node<V, T>* n=heap;
		if(n==NULL)return NULL;
		do {
			if(n->key==key)return n;
			node<V, T>* ret=_find(n->child,key);
			if(ret)return ret;
			n=n->next;
		}while(n!=heap);
		return NULL;
	}
};

#endif