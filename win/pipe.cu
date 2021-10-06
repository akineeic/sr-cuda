#include "pipe.cuh"



FILE * stdinf;
FILE * stdoutf;

timed_mutex readlock[CNT];
timed_mutex writelock[CNT];
timed_mutex wlock[CNT];
timed_mutex gpulock[CNT];
timed_mutex threadlock[CNT];
timed_mutex donelock[CNT];

my_sr SR[CNT];

thread sr_thread[CNT];
int tnow = 0;
bool active;
int get_len(char * input)
{
	int res = 0;
	while (1)
	{
		if (input[res] == '\0')
		{
			return res;
		}
		res++;
	}
}

void write_std()
{
	while (active)
	{
		while (1 != wlock[tnow].try_lock_for(chrono::milliseconds(300)))
		{
			if (!active)
			{
				break;
			}
			continue;
		}
		fwrite("FRAME\n", 1, 6, stdout);
		fwrite(&SR[tnow].out_YUV[0], 1, 3840 * 2160 * 3 / 2, stdout);
		tnow = (tnow + 1) % CNT;
		writelock[tnow].unlock();
	}
	
}

void loop(int tid)
{
	//cout1 << "Thread startec.\n";
	while (true)
	{
		while (1 != threadlock[tid].try_lock_for(chrono::milliseconds(300)))
		{
			
			continue;
		}
		if (!active)
		{
			//cout1 << "Thread stopping.\n";
			break;
		}

		while (1 != gpulock[tid].try_lock_for(chrono::milliseconds(300)))
		{
			
			continue;
		}

		donelock[tid].lock();
		SR[tid].super_resolution();
		readlock[(tid + 1) % CNT].unlock();
		SR[tid].sync();

		gpulock[(tid + 1) % CNT].unlock();

		while (1 != writelock[tid].try_lock_for(chrono::milliseconds(300)))
		{
			
			continue;
		}
		SR[tid].copy_out();
		SR[tid].sync();
		wlock[tid].unlock();
		
		donelock[tid].unlock();


	}
	//cout1 << "Thread stoppec.\n";
}

void init()
{
	_setmode(_fileno(stdout), _O_BINARY);
	_setmode(_fileno(stdin), _O_BINARY);
	//stdinf = fdopen(dup(fileno(stdin)), "wb");
	//stdoutf = fdopen(dup(fileno(stdout)), "wb");
	
	active = true;

	
	for (int i = 0; i < CNT; i++)
	{
		SR[i].init();
		readlock[i].lock();
		threadlock[i].lock();
		writelock[i].lock();
		gpulock[i].lock();
		SR[i].sync();
		wlock[i].lock();
	}
	writelock[0].unlock();
	gpulock[0].unlock();
	//cout1 << "---------starting threads: totally " << CNT << ".---------" << endl;
	for (int i = 0; i < CNT; i++)
	{
		sr_thread[i] = thread(loop, i);
		sr_thread[i].detach();
	}
	thread watch(write_std);
	watch.detach();
	Sleep(10);
}

void free()
{
	//cout1 << "---------stopping threads---------" << endl;
	for (int i = 0; i < CNT; i++)
	{
		while (1 != donelock[i].try_lock_for(chrono::milliseconds(300)))
		{
			continue;
		}
	}
	active = false;
	for (int i = 0; i < CNT; i++)
	{
		threadlock[i].unlock();
	}

	for (int i = 0; i < CNT; i++)
	{
		SR[i].free();
	}
	cudaDeviceReset();

	//usleep(1000);
}

int main()
{
	
	cudaSetDevice(0);
	//cudaSetDeviceFlags(cudaDeviceScheduleSpin);
	init();

	int process_id = 0;
	char * y4mheader = "YUV4MPEG2 C420 W3840 H2160 F60:1 Ip A0:0\n";
	fwrite(y4mheader, 1, get_len(y4mheader), stdout);
	char ccc;
	while (true)
	{
		if (fread(&ccc, 1, 1, stdin) == 0)
		{
			free();
			return -1;
		}
		if (ccc == '\n')
		{
			break;
		}
	}

	char * cc = new char[7];
	while (fread(cc, 1, 6, stdin)>0)
	{

		fread(&SR[process_id].in_YUV[0], 1, 1920 * 1080 * 3 / 2, stdin);
		threadlock[process_id].unlock();
		process_id = (process_id + 1) % CNT;
		while (1 != readlock[process_id].try_lock_for(chrono::milliseconds(300)))
		{
			
			continue;
		}
	}
	free();
	fflush(stdout);
	return 0;
}
