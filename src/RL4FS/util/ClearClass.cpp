#include "RL4FS/util/ClearClass.h"
#include "RL4FS/util/Tools.h"

namespace RL4FS
{
	void ClearClass::clear()
	{
		if (Property::getProperty()->getDistributedNodes() > 1)
		{
#ifdef USE_MPICH
			MPI_Finalize();
#else
			LOG(ERROR) << "ClearClass Please USE_MPICH = ON when CMake this project, or you cannot set Property::getProperty()->setDistributedNodes() > 1";
			exit(-1);
#endif
		}
		delete Property::m_instance;
		delete globalVar::m_instance;
	}
}