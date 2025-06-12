#include <iostream>
#include <cmath>
#include <vector>
#include <mpi.h>
#include <algorithm>
#include <array>
#include <fstream> 


// ** PARTICLES ** 
struct Particle {
    double px;
    double py;
    double vx; 
    double vy; 
    double Q = -1.0; 
    double m = 1.0; 
};

struct ParticleData {
    int step;
    double px;
    double py;
    int has_particle;
};

int main(int argc, char** argv) {

double Lx = 8.0 * M_PI;
double Ly = 4.0 * M_PI;

int NGx = 128; 
int NGy = 64;

double dx = Lx / NGx; 
double dy = Ly / NGy;

double dt = 0.5;
int NT = 500; 

MPI_Init(&argc, &argv);

int global_rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

int dims[2] = {2, 2};
int periods[2] = {1, 1};
int reorder = 0;
MPI_Comm cart_comm;

MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);

int rank; 
MPI_Comm_rank(cart_comm, &rank);
int coords[2];

MPI_Cart_coords(cart_comm, rank, 2, coords);

int up; 
int down; 
int left; 
int right;
MPI_Cart_shift(cart_comm, 0, 1, &left, &right); 
MPI_Cart_shift(cart_comm, 1, 1, &down, &up);  

double x0, xF, y0, yF; 
double domainLengthx = Lx / dims[0]; 
double domainLengthy = Ly / dims[1];

x0 = coords[0] * domainLengthx; 
xF = (coords[0] + 1) * domainLengthx;
y0 = coords[1] * domainLengthy; 
yF = (coords[1] + 1) * domainLengthy;

std::vector<Particle> particles;
bool has_particle;
if (rank == 1) {
    Particle p;
    p.px = (domainLengthx / 2) - 5 * dx; 
    p.py = yF - 5 * dy; 
    p.vx = M_PI;
    p.vy = 0.0; 
    particles.push_back(p);
    has_particle = true; 
}
else { 
    has_particle = false;
}

double B_z = 0.5;
double Fb_x = particles[0].Q * particles[0].vy * B_z; 
double Fb_y = -particles[0].Q * particles[0].vx * B_z;

if (has_particle == true) {
    particles[0].vx = particles[0].vx + (Fb_x * (dt / 2));
    particles[0].vy = particles[0].vy + (Fb_y * (dt / 2));
}

std::vector<double> send_left(4);
std::vector<double> send_right(4);
std::vector<double> send_up(4);
std::vector<double> send_down(4);

std::vector<double> recv_left(4);
std::vector<double> recv_right(4);
std::vector<double> recv_up(4);
std::vector<double> recv_down(4);

for (int t = 0; t < NT; ++t) {

    send_left.assign(4, 0.0);
    send_right.assign(4, 0.0);
    send_up.assign(4, 0.0);
    send_down.assign(4, 0.0);

    recv_left.assign(4, 0.0);
    recv_right.assign(4, 0.0);
    recv_up.assign(4, 0.0);
    recv_down.assign(4, 0.0);

    if (has_particle == true) {
        particles[0].px += (particles[0].vx * dt);
        particles[0].py += (particles[0].vy * dt);

        double omega_c = particles[0].Q * B_z / particles[0].m;
        double theta = omega_c * dt;

        double vx_old = particles[0].vx;
        double vy_old = particles[0].vy;

        particles[0].vx = vx_old * cos(theta) - vy_old * sin(theta);
        particles[0].vy = vx_old * sin(theta) + vy_old * cos(theta); 
    }
    
    if (has_particle == true && particles[0].px < x0) {
        send_left[0] = particles[0].px;
        send_left[1] = particles[0].py;
        send_left[2] = particles[0].vx;
        send_left[3] = particles[0].vy;
    
        has_particle = false;
    }

    MPI_Sendrecv(send_left.data(), 4, MPI_DOUBLE, left, 0, recv_left.data(), 4, MPI_DOUBLE, right, 0, cart_comm, MPI_STATUS_IGNORE);

    if (recv_left[0] != 0.0 && recv_left[1] != 0.0) {
        Particle p;
        p.px = recv_left[0];
        p.py = recv_left[1];
        p.vx = recv_left[2];
        p.vy = recv_left[3];
        particles.push_back(p);
        has_particle = true;
    }

    if (has_particle == true && particles[0].px >= xF) {
        send_right[0] = particles[0].px;
        send_right[1] = particles[0].py;
        send_right[2] = particles[0].vx;
        send_right[3] = particles[0].vy;

        has_particle = false;
    }

    MPI_Sendrecv(send_right.data(), 4, MPI_DOUBLE, right, 1, recv_right.data(), 4, MPI_DOUBLE, left, 1, cart_comm, MPI_STATUS_IGNORE);

    if (recv_right[0] != 0.0 && recv_right[1] != 0.0) {
        Particle p;
        p.px = recv_right[0];
        p.py = recv_right[1];
        p.vx = recv_right[2];
        p.vy = recv_right[3];
        particles.push_back(p);
        has_particle = true;
    }

    if (has_particle == true && particles[0].py < y0) {
        send_down[0] = particles[0].px;
        send_down[1] = particles[0].py;
        send_down[2] = particles[0].vx;
        send_down[3] = particles[0].vy;

        has_particle = false;
    }

    MPI_Sendrecv(send_down.data(), 4, MPI_DOUBLE, down, 2, recv_down.data(), 4, MPI_DOUBLE, up, 2, cart_comm, MPI_STATUS_IGNORE);

    if (recv_down[0] != 0.0 && recv_down[1] != 0.0) {
        Particle p;
        p.px = recv_down[0];
        p.py = recv_down[1];
        p.vx = recv_down[2];
        p.vy = recv_down[3];
        particles.push_back(p);
        has_particle = true;
    }

    if (has_particle == true && particles[0].py >= yF) {
        send_up[0] = particles[0].px;
        send_up[1] = particles[0].py;
        send_up[2] = particles[0].vx;
        send_up[3] = particles[0].vy;

        has_particle = false;
    }

    MPI_Sendrecv(send_up.data(), 4, MPI_DOUBLE, up, 3, recv_up.data(), 4, MPI_DOUBLE, down, 3, cart_comm, MPI_STATUS_IGNORE);

    if (recv_up[0] != 0.0 && recv_up[1] != 0.0) {
        Particle p;
        p.px = recv_up[0];
        p.py = recv_up[1];
        p.vx = recv_up[2];
        p.vy = recv_up[3];
        particles.push_back(p);
        has_particle = true;
    }

    ParticleData local_data;
    local_data.step = t;
    if (has_particle) {
        local_data.px = particles[0].px;
        local_data.py = particles[0].py;
        local_data.has_particle = 1;
    }   
    else {
        local_data.px = 0.0;
        local_data.py = 0.0;
        local_data.has_particle = 0;
    }

    std::vector<ParticleData> all_data;
    if (rank == 0) all_data.resize(size);

    MPI_Gather(&local_data, sizeof(ParticleData), MPI_BYTE, all_data.data(), sizeof(ParticleData), MPI_BYTE, 0, cart_comm);

    if (rank == 0) {
        static std::ofstream outfile("particle_positions.txt", std::ios::out);
        if (!outfile.is_open()) {
            std::cerr << "Error opening file!" << std::endl;
            MPI_Abort(cart_comm, 1);
        }
        for (int r = 0; r < size; ++r) {
            if (all_data[r].has_particle) {
                outfile << all_data[r].step << " "
                    << all_data[r].px << " "
                    << all_data[r].py << "\n";
            }   
        }
        outfile.flush();  // make sure data is written every step
    }
}

MPI_Finalize(); 
return 0;
}

// mpic++ -O2 -std=c++17 cyclotron.cpp -o cyclotron