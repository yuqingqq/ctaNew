// Research-only branch execution, with parity checks against the frozen Python lane.
// Each public leg is applied once to a whole portfolio in price/time order.
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

static double shown(const double* f, int side, double price, double delta) {
    // frame: venue time, received time, bid prices/sizes, ask prices/sizes
    int p = 2 + side * 10;
    double last = f[p+4];
    if (last >= 0 && (side == 0 ? price < last+delta : price > last+delta)) return NAN;
    double q = 0;
    for (int j=0;j<5;j++) if (f[p+j]>=0 && f[p+j]+delta==price) q+=f[p+5+j];
    return q;
}

extern "C" int evaluate(int ns, int nb, int no, const long long* fo,
 const long long* to, const double* frames, const double* trades,
 const double* shifts, const double* endmid, const double* orders,
 const double* action, double position, double startmid, double horizon,
 double place_ms, double cancel_ms, double pending_cancel_ms, double hold_ticks,
 double* output, double* quantities) {
    // orders: side,rung,price,remaining,ahead,activation,removal,submit,new-order
    if(ns<1 || nb<1 || no<0 || no>72 || horizon<=0) return 1;
    std::vector<int> priority(no); std::iota(priority.begin(),priority.end(),0);
    std::stable_sort(priority.begin(),priority.end(),[&](int a,int b){
        const double *x=orders+a*9,*y=orders+b*9;
        if(x[0]!=y[0]) return x[0]<y[0];
        if(x[2]!=y[2]) return x[0]==0 ? x[2]>y[2] : x[2]<y[2];
        return x[5]<y[5];
    });
    for(int s=0;s<ns;s++) {
        int nf=fo[s+1]-fo[s], nt=to[s+1]-to[s];
        if(nf<1) return 2;
        const double* f=frames+fo[s]*22; const double* tape=trades+to[s]*4;
        double delta=shifts[s];
        for(int b=0;b<nb;b++) {
            std::vector<double> rem(no),ahead(no),remove(no),got(no,0.);
            std::vector<int> ci(no,0);
            double pos=position,cash=0,total=0,bq=0,aq=0;
            for(int i=0;i<no;i++) {
                const double* o=orders+i*9; int act=(int)action[b*no+i];
                rem[i]=act==0?0:o[3]; ahead[i]=std::isfinite(o[4])?o[4]:0.; remove[i]=o[6];
                if(act==2) remove[i]=std::min(remove[i],o[5]>0 ? std::max(cancel_ms,o[7]+pending_cancel_ms):cancel_ms);
                if(o[8]>0 && rem[i]>0) {
                    int j=0; while(j+1<nf && f[(j+1)*22]<=o[5]) j++;
                    double local_delta=j==0?0.:delta;
                    double opp=f[j*22+2+(1-(int)o[0])*10];if(opp>=0)opp+=local_delta;
                    if(o[0]==0 ? (0<=opp && opp<=o[2]) : opp>=o[2]) {rem[i]=0;continue;}
                    double a=shown(f+j*22,(int)o[0],o[2],local_delta);
                    ahead[i]=std::isfinite(a)?a:0.;ci[i]=j;
                }
            }
            int it=0,known=0;
            for(double grid=0.;grid<=horizon+50.;grid+=50.) {
                double limit=std::min(grid,horizon);
                while(it<nt && tape[it*4]<=limit) {
                    const double* tr=tape+it*4;double t=tr[0],p=tr[2]+delta,left=tr[3],used=0.;int side=(int)tr[1];
                    for(int i:priority) {
                        const double* o=orders+i*9;
                        if(o[0]!=side || rem[i]<=1e-12 || o[5]>t || t>=remove[i]) continue;
                        int j=ci[i];
                        while(j+1<nf && f[(j+1)*22]<t) {
                            j++;double a=shown(f+j*22,side,o[2],j==0?0.:delta);
                            if(std::isfinite(a)) ahead[i]=std::min(ahead[i],a);
                        }
                        ci[i]=j;
                        bool through=side==0?p<o[2]:p>o[2];double q=0.;
                        if(p==o[2]) {double use=std::min(left,ahead[i]);ahead[i]-=use;q=std::min(left-use,rem[i]);}
                        else if(through) {q=std::min(left,rem[i]);ahead[i]=0.;}
                        if(q>1e-12) {left-=q;used+=q;rem[i]-=q;got[i]+=q;total+=q;
                            double sign=side==0?1.:-1.;pos+=sign*q;cash-=sign*q*o[2];
                            if(side==0)bq+=q;else aq+=q;
                        }
                    }
                    if(used>tr[3]+1e-7) return 3;
                    it++;
                }
                if(grid>=horizon) break;
                while(known+1<nf && f[(known+1)*22+1]<=grid) known++;
                for(int i=0;i<no;i++) {
                    const double* o=orders+i*9;
                    if(rem[i]<=1e-12 || remove[i]<=grid || (grid==0 && o[8]>0)) continue;
                    double best=f[known*22+2+(int)o[0]*10],opp=f[known*22+2+(1-(int)o[0])*10];
                    double local_delta=known==0?0.:delta;
                    if(best>=0)best+=local_delta;if(opp>=0)opp+=local_delta;
                    double behind=o[0]==0?best-o[2]:o[2]-best;
                    bool cross=o[0]==0?o[2]>=opp:o[2]<=opp;
                    if(best<0 || behind>hold_ticks+o[1] || cross) {
                        double arrival=grid<o[5]?std::max(grid+cancel_ms,o[7]+pending_cancel_ms):grid+cancel_ms;
                        remove[i]=std::min(remove[i],arrival);
                    }
                }
            }
            double* out=output+(s*nb+b)*6;
            out[0]=cash+pos*endmid[s]-position*startmid;
            out[1]=total;out[2]=pos;out[3]=cash;out[4]=bq;out[5]=aq;
            for(int i=0;i<no;i++) {
                if(got[i]>orders[i*9+3]+1e-7) return 4;
                quantities[(s*nb+b)*no+i]=got[i];
            }
        }
    }
    return 0;
}
